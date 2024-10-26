import base64
import json
import time
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken
from huggingface_hub import snapshot_download

CHAT_INIT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: {date}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>"""

CHAT_CONT = """<|start_header_id|>user<|end_header_id|>

{text}<|eot_id|>"""

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
        freqs = pids[:, None, :, None] @ mx.repeat(self._inv_freq[None, None, None, :], pids.shape[0], axis=0)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        return mx.cos(emb), mx.sin(emb)

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
        q, k = self.apply_rope(q, k, *rope)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        w = (q * self.scale) @ mx.repeat(k, self.n_repeat, axis=1).transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ mx.repeat(v, self.n_repeat, axis=1)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(dtype), (k,v)
    @staticmethod
    @mx.compile
    def apply_rope(q, k, cos, sin):
        q1, q2 = mx.split(q, 2, axis=-1)
        rq = mx.concatenate([-q2, q1], axis = -1)
        k1, k2 = mx.split(k, 2, axis=-1)
        rk = mx.concatenate([-k2, k1], axis = -1)
        return (q * cos + rq * sin), (k * cos + rk * sin)

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
        self._rope = RoPE(cfg)
        if cfg['tie_word_embeddings']:
            self.tie = True
        else:
            self.tie = False
            self.lm_head = nn.Linear(cfg['hidden_size'], cfg['vocab_size'], bias=False)
    def __call__(self, toks, pids, mask, cache):
        rope = self._rope(pids)
        out, cache = self.model(toks=toks, rope=rope, mask=mask, cache=cache)
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
        model.load_weights(f'{path_hf}/llama_32_1b_it_model.safetensors', strict=True)
        model.eval()
        mx.eval(model)
        self.model = model
        self.tokenizer = Tokenizer(f'{path_hf}/llama_32_1b_it.tiktoken')
        self.num_layers = cfg['num_hidden_layers']
        self.cache = None
        self.mask = None
        self.pids = 0
    def __call__(self, inputs, max_new=500, verbose=True):
        tic = time.perf_counter()
        if isinstance(inputs, str):
            inputs = [inputs]
        assert len(inputs) == 1, 'Batching is not implemented yet'
        chat_fmt = CHAT_INIT if self.cache is None else CHAT_CONT
        chat_fmt = chat_fmt.format(date=datetime.now().strftime('%d %b %Y'), text='{text}')
        inputs = self.tokenizer.encode([chat_fmt.format(text=i) for i in inputs])
        toks, pids, mask = self.pad_to_batch(inputs)
        cache = [None] * self.num_layers if self.cache is None else self.cache
        result = mx.zeros((toks.shape[0],0), dtype=mx.uint32)
        goon = mx.ones((toks.shape[0],1), dtype=mx.bool_)
        for _ in range(max_new):
            toks, cache = self.model(toks=toks, pids=pids, mask=self.get_mask(mask, toks.shape[-1]), cache=cache)
            toks = mx.argmax(toks[:,-1,:], axis=-1, keepdims=True)
            pids = pids[:,-1:]+1
            mx.eval(toks, pids, mask, cache)
            result = mx.concatenate([result, toks], axis=-1)
            goon *= (toks != 128009) # <|eot_id|>
            if goon.sum() < 1:
                break
            mask = mx.pad(mask, ((0,0), (0, 1)), constant_values=1)
        self.cache = cache
        self.mask = mask
        self.pids = pids
        text = self.tokenizer.decode(result.tolist())
        if verbose:
            tic = time.perf_counter()-tic
            num = result.size
            tps = num/tic
            print(f'{self.tokenizer.decode(inputs)}\n\n---\n\n{'\n\n---\n\n'.join(text)}\n\n---\n\n{tps:.2f} tps ({num} in {tic:.2f} seconds)')
        return text
    def pad_to_batch(self, input_ids):
        max_length = max(len(sublist) for sublist in input_ids)
        toks = mx.array([[128009]*(max_length-len(sublist)) + sublist for sublist in input_ids])
        pids = mx.array([[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids])
        mask = mx.array([[False]*(max_length-len(sublist)) + [True]*len(sublist) for sublist in input_ids])
        if self.cache is not None:
            pids += self.pids
            mask = mx.concatenate([self.mask, mask], axis=-1)
        return toks, pids, mask
    @staticmethod
    @mx.compile
    def get_mask(mask, trunc):
        _mask = mx.triu(mx.full((mask.shape[-1], mask.shape[-1]), False), k=1)
        _mask *= mx.where(mask[:, None, :, None]*mask[:, None, None, :], 0, -mx.inf)
        _mask = _mask[...,-trunc:,:]
        return _mask
chat = Chat()
chat("What's the weather like in Busan?")
chat("What's it like right now?")
chat("Temperature")

# ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 25 Oct 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the weather like in Busan?<|eot_id|>"]

# ---

# <|start_header_id|>assistant<|end_header_id|>

# In Busan, South Korea, the weather is typically warm and humid throughout the year. Here's a breakdown of the typical weather conditions:

# * Summer (June to August): Hot and humid, with average high temperatures ranging from 28°C (82°F) to 32°C (90°F). Overnight temperatures can still be quite warm, around 22°C (72°F) to 25°C (77°F).
# * Autumn (September to November): Mild and pleasant, with average high temperatures ranging from 20°C (68°F) to 25°C (77°F). Overnight temperatures can still be quite comfortable, around 15°C (59°F) to 20°C (68°F).
# * Winter (December to February): Cool and wet, with average high temperatures ranging from 10°C (50°F) to 15°C (59°F). Overnight temperatures can drop significantly, around 5°C (41°F) to 10°C (50°F).
# * Spring (March to May): Mild and pleasant, with average high temperatures ranging from 15°C (59°F) to 20°C (68°F). Overnight temperatures can still be quite comfortable, around 10°C (50°F) to 15°C (59°F).

# It's worth noting that Busan can experience occasional extreme weather conditions, such as typhoons, heatwaves, and cold snaps. It's always a good idea to check the weather forecast before traveling to Busan.

# In addition to the general weather patterns, Busan can also experience:

# * Humidity: High humidity is common in Busan, especially during the summer months.
# * Rainfall: Busan can experience heavy rainfall, especially during the summer months.
# * Thunderstorms: Thunderstorms are common in Busan, especially during the summer months.
# * Fog: Fog can occur in Busan, especially in the mornings and evenings.

# Overall, Busan's weather is characterized by warm and humid conditions throughout the year, with occasional extreme weather conditions.<|eot_id|>

# ---

# 65.76 tps (406 in 6.17 seconds)
# ["<|start_header_id|>user<|end_header_id|>\n\nWhat's it like right now?<|eot_id|>"]

# ---

# <|start_header_id|>assistant<|end_header_id|>

# I don't have real-time access to current weather conditions. However, I can suggest some ways for you to find out the current weather in Busan, South Korea.

# 1. Check online weather websites: You can check websites like AccuWeather, Weather.com, or the Korea Meteorological Administration (KMA) for the current weather conditions in Busan.
# 2. Use a mobile app: You can download mobile apps like Dark Sky, Weather Underground, or The Weather Channel to get the current weather conditions in Busan.
# 3. Check social media: You can check social media platforms like Twitter or Facebook for updates on the current weather in Busan.

# Please note that the weather in Busan can change quickly, so it's always a good idea to check multiple sources for the most up-to-date information.

# If you're looking for a specific type of weather, such as a specific temperature or precipitation, I can try to help you with that. Just let me know what you're looking for, and I'll do my best to provide you with the information you need.<|eot_id|>

# ---

# 61.13 tps (220 in 3.60 seconds)
# ['<|start_header_id|>user<|end_header_id|>\n\nTemperature<|eot_id|>']

# ---

# <|start_header_id|>assistant<|end_header_id|>

# I can provide you with general information about the temperature in Busan, South Korea.

# As I mentioned earlier, Busan has a humid subtropical climate, with warm and humid summers, mild and pleasant winters, and a moderate climate throughout the year.

# **Summer (June to August)**

# * Average high temperature: 28°C (82°F)
# * Average low temperature: 22°C (72°F)

# **Autumn (September to November)**

# * Average high temperature: 20°C (68°F)
# * Average low temperature: 15°C (59°F)

# **Winter (December to February)**

# * Average high temperature: 10°C (50°F)
# * Average low temperature: 5°C (41°F)

# **Spring (March to May)**

# * Average high temperature: 15°C (59°F)
# * Average low temperature: 10°C (50°F)

# Please note that these are general temperature ranges, and actual temperatures can vary from year to year.<|eot_id|>

# ---

# 58.44 tps (203 in 3.47 seconds)
