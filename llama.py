import json
import time
import base64
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
import tiktoken

class Tokenizer:
    def __init__(self, path_tok):
        with open(path_tok) as f:
            ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line)}
        n_vocab = len(ranks)
        specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>", *[f"<|reserved_special_token_{2 + i}|>" for i in range(245)]]
        special_tokens = {k:(n_vocab+i) for i,k in enumerate(specials)}
        self.encoding = tiktoken.Encoding(name='jj', explicit_n_vocab=n_vocab + len(special_tokens), pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", mergeable_ranks=ranks, special_tokens=special_tokens)
    def encode(self, lot):
        if isinstance(lot, str):
            lot = [lot]
        return [self.encoding.encode(t, allowed_special='all') for t in lot]
    def decode(self, lol):
        if isinstance(lol[0], int):
            lol = [lol]
        return [self.encoding.decode(l) for l in lol]

CHAT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 25 Oct 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>"""

class RoPE:
    def __init__(self, cfg):
        dim, base, factor, low, high, omax = cfg['head_dim'], cfg['rope_theta'], cfg['rope_scaling']['factor'], cfg['rope_scaling']['low_freq_factor'], cfg['rope_scaling']['high_freq_factor'], cfg['rope_scaling']['original_max_position_embeddings']
        low_len, high_len = omax/low, omax/high
        freqs = (base ** (np.arange(0, dim, 2) / dim))
        wavelens = 2 * np.pi * freqs
        freqs = np.where(wavelens > low_len, freqs*factor, freqs)
        between = (omax/wavelens - low) / (high - low)
        between = freqs / ((1 - between) / factor + between)
        freqs = np.where((low_len < wavelens) & (wavelens < high_len), between, freqs)
        self._inv_freq = mx.array(1 / freqs)
    def __call__(self, pids):
        freqs = (pids[:, None, :, None] @ mx.repeat(self._inv_freq[None, None, None, :], pids.shape[0], axis=0))
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos, sin

@mx.compile
def apply_rope(q, k, cos, sin):
    q1, q2 = mx.split(q, 2, axis=-1)
    rq = mx.concatenate([-q2, q1], axis = -1)
    k1, k2 = mx.split(k, 2, axis=-1)
    rk = mx.concatenate([-k2, k1], axis = -1)
    return (q * cos + rq * sin), (k * cos + rk * sin)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['hidden_size']
        self.n_heads = n_heads = cfg['num_attention_heads']
        self.n_kv_heads = n_kv_heads = cfg['num_key_value_heads']
        self.n_repeat = int(n_heads / n_kv_heads)
        self.head_dim = head_dim = cfg['head_dim']
        self.scale = head_dim**-0.5
        if hasattr(cfg, "attention_bias"):
            attention_bias = cfg['attention_bias']
        else:
            attention_bias = False
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)
    def __call__(self, x, rope, mask, cache):
        B, L, D = x.shape
        dtype = x.dtype
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, k, *rope)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        w = (q * self.scale) @ mx.repeat(k, self.n_repeat, axis=1).transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ mx.repeat(v, self.n_repeat, axis=1)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(dtype), (k,v)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg['hidden_size']
        hidden_dim = cfg['intermediate_size']
        mlp_bias = cfg['mlp_bias']
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg['hidden_size'], eps=cfg['rms_norm_eps'])
        self.post_attention_layernorm = nn.RMSNorm(cfg['hidden_size'], eps=cfg['rms_norm_eps'])
    def __call__(self, x, rope, mask, cache):
        r, cache = self.self_attn(self.input_layernorm(x), rope=rope, mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache

class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg['vocab_size'], cfg['hidden_size'])
        self.layers = [TransformerBlock(cfg=cfg) for _ in range(cfg['num_hidden_layers'])]
        self.norm = nn.RMSNorm(cfg['hidden_size'], eps=cfg['rms_norm_eps'])
    def __call__(self, toks, rope, mask, cache):
        h = self.embed_tokens(toks)
        for i, l in enumerate(self.layers):
            h, cache[i] = l(h, rope=rope, mask=mask, cache=cache[i])
        return self.norm(h), cache

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = LlamaModel(cfg)
        self.num_layers = cfg['num_hidden_layers']
        self._rope = RoPE(cfg)
        if cfg['tie_word_embeddings']:
            self.tie = True
        else:
            self.tie = False
            self.lm_head = nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)
    def __call__(self, toks, pids, mask, cache=None):
        if cache is None:
            cache = [None] * self.num_layers
            offset = 0
        else:
            offset = cache[0][0].shape[-2]
        len_in = toks.shape[-1]
        _mask = mx.triu(mx.full((offset+len_in, offset+len_in), -mx.inf), k=1)
        if mask is not None:
            _mask += mx.where(mask[:, None, :, None]*mask[:, None, None, :]==1, 0, -mx.inf)
        _mask = _mask[...,-len_in:,:]
        rope = self._rope(pids)
        out, cache = self.model(toks=toks, rope=rope, mask=_mask, cache=cache)
        if self.tie:
            return self.model.embed_tokens.as_linear(out), cache
        return self.lm_head(out), cache
    @property
    def layers(self):
        return self.model.layers

class Chat:
    def __init__(self):
        path_hf = snapshot_download(repo_id='JosefAlbers/llama', allow_patterns=["llama_32_1b_it*"])
        with open(f'{path_hf}/llama_32_1b_it_config.json', 'r') as f:
            cfg = json.load(f)
        model = Model(cfg)
        model.load_weights(f'{path_hf}/llama_32_1b_it_model.safetensors', strict=False)
        model.eval()
        mx.eval(model)
        self.model = model
        self.tokenizer = Tokenizer(f'{path_hf}/llama_32_1b_it.tiktoken')
    def __call__(self, inputs, max_new=500, verbose=True):
        tic = time.perf_counter()
        if isinstance(inputs, str):
            inputs = [inputs]
        assert len(inputs) == 1, 'Batching is not implemented yet'
        inputs = self.tokenizer.encode([CHAT.format(text=i) for i in inputs])
        toks, pids, mask = self.pad_to_batch(inputs)
        cache = None
        result = mx.zeros((toks.shape[0],0), dtype=mx.uint32)
        goon = mx.ones((toks.shape[0],1), dtype=mx.bool_)
        for _ in range(max_new):
            toks, cache = self.model(toks=toks, pids=pids, mask=mask, cache=cache)
            toks = mx.argmax(toks[:,-1,:], axis=-1, keepdims=True)
            pids = pids[:,-1:]+1
            mask = mx.pad(mask, ((0,0), (0, 1)), constant_values=1)
            mx.eval(toks, pids, mask)
            result = mx.concatenate([result, toks*goon], axis=-1)
            goon *= (toks != 128009) # <|eot_id|>
            if goon.sum() < 1:
                break
        text = self.tokenizer.decode(result.tolist())
        if verbose:
            tic = time.perf_counter()-tic
            num = result.size
            tps = num/tic
            print(f'{self.tokenizer.decode(inputs)}\n\n---\n\n{'\n\n---\n\n'.join(text)}\n\n---\n\n{tps:.2f} tps ({num} in {tic:.2f} seconds)')
        return text
    @staticmethod
    def pad_to_batch(input_ids):
        max_length = max(len(sublist) for sublist in input_ids)
        toks = mx.array([[0]*(max_length-len(sublist)) + sublist for sublist in input_ids])
        pids = mx.array([[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids])
        mask = mx.array([[0]*(max_length-len(sublist)) + [1]*len(sublist) for sublist in input_ids])
        return toks, pids, mask

chat = Chat()
chat("What's the weather right now in Busan?")

# ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 25 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the weather right now in Busan?<|eot_id|>"]

# ---

# <|start_header_id|>assistant<|end_header_id|>

# However, I'm a large language model, I don't have real-time access to current weather conditions. But I can suggest some ways for you to find out the current weather in Busan.

# You can:

# 1. Check online weather websites or apps, such as AccuWeather, Weather.com, or the Korea Meteorological Administration (KMA) website.
# 2. Use a mobile app like Dark Sky or Weather Underground, which provide current weather conditions and forecasts for Busan.
# 3. Tune into local news or radio stations, which often provide weather updates.

# Please note that the weather can change rapidly, so it's always a good idea to check multiple sources for the most up-to-date information.

# If you'd like, I can also provide you with general information about the climate and weather patterns in Busan.<|eot_id|>

# ---

# 65.66 tps (168 in 2.56 seconds)
