

# nanollama32

A compact and efficient implementation of the Llama 3.2 in a single file, featuring minimal dependencies—**no transformers library required, even for tokenization**.

## Overview

`nanollama32` provides a lightweight and straightforward implementation of the Llama model. It features:

- Minimal dependencies
- Easy-to-use interface
- Efficient performance suitable for various applications

## Installation

To get started, clone this repository and install the necessary packages. 

```bash
git clone https://github.com/JosefAlbers/nanollama32.git
cd nanollama32
pip install -r requirements.txt
```

## Usage

Here’s a quick example of how to use `nanollama32`:

```python
from nanollama32 import Chat

chat = Chat()
response = chat("What's the weather like in Busan?")
print(response)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project builds upon the [MLX implementation](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py) and [Karpathy's LLM.c implementation](https://github.com/karpathy/llm.c/blob/master/train_llama3.py) of the Llama model. Special thanks to the contributors of both projects for their outstanding work and inspiration.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
