# nanollama32

A compact and efficient implementation of the Llama 3.2 in a single file, featuring minimal dependencies—**no transformers library required, even for tokenization**.

## Overview

`nanollama32` provides a lightweight and straightforward implementation of the Llama model. It features:

- Minimal dependencies
- Easy-to-use interface
- Efficient performance suitable for various applications

## Quick Start

To get started, clone this repository and install the necessary packages. 

```zsh
pip install nanollama
```

Here’s a quick example of how to use `nanollama32`:

```python
>>> from nanollama32 import Chat

# Initialize the chat instance
>>> chat = Chat()

# Start a conversation
>>> chat("What's the weather like in Busan?")
# Llama responds with information about the weather

# Follow-up question that builds on the previous context
>>> chat("And how about the temperature?")
# Llama responds with the temperature, remembering the previous context

# Another follow-up, further utilizing context
>>> chat("What should I wear?")
# Llama suggests clothing based on the previous responses
```

## Command-Line Interface

You can also run `nanollama32` from the command line:

```zsh
nlm how to create a new conda env
# Llama responds with ways to create a new conda environment and prompts the user for further follow-up questions
```

### Managing Chat History

- **--history**: Specify the path to the JSON file where chat history will be saved and/or loaded from. If the file does not exist, a new one will be created.
- **--resume**: Use this option to resume the conversation from a specific point in the chat history.

For example, you can specify `0` to resume from the most recent entry:

```zsh
nlm "and to list envs?" --resume 0
```

Or, you can resume from a specific entry in history:

```zsh
nlm "and to delete env?" --resume 20241026053144
```

### Adding Text from Files

You can include text from any number of external files by using the `{...}` syntax in your input. For example, if you have a text file named `langref.rst`, you can include its content in your input like this:

```zsh
nlm to create reddit bots {langref.rst}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

This project builds upon the [MLX implementation](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py) and [Karpathy's LLM.c implementation](https://github.com/karpathy/llm.c/blob/master/train_llama3.py) of the Llama model. Special thanks to the contributors of both projects for their outstanding work and inspiration.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.