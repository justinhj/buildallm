# Project Context: buildallm

## Project Overview
`buildallm` appears to be a Python project focused on building the foundational components of a Large Language Model (LLM), starting with text tokenization. The current codebase implements custom tokenizers that process raw text, build a vocabulary, and handle encoding (text to integers) and decoding (integers to text).

## Key Technologies
*   **Language:** Python (Requires >= 3.12)
*   **Package Manager:** `uv` (indicated by `uv.lock`)
*   **Libraries:** Uses the standard `re` library for regex-based splitting and tokenization.

## Code Structure
*   `main.py`: The entry point. It contains:
    *   `SimpleTokenizerV1`: A basic tokenizer class.
    *   `SimpleTokenizerV2`: An improved tokenizer that handles unknown tokens (`<|unk|>`) and extended punctuation.
    *   `main()`: Reads `./data/the-verdict.txt`, demonstrates tokenization, vocabulary creation, and the encode/decode cycle.
*   `data/`: Directory containing input text data (`the-verdict.txt`).
*   `pyproject.toml`: Project configuration and metadata.

## Building and Running

### Prerequisites
*   Python 3.12 or higher.
*   `uv` (recommended for dependency management).

## Adding Python packages
* When the user says install a package use `uv add <package name>` do not use pip install.

### Execution
To run the main script:

```bash
uv run main.py
```
Or with standard Python:
```bash
python main.py
```

## Development Conventions
*   **Tokenization:** The project uses regular expressions for splitting text into tokens, preserving punctuation as separate tokens.
*   **Vocabulary:** Vocabulary is built dynamically from the input text file.
*   **Special Tokens:** Support for `<|unk|>` (unknown) and `<|endoftext|>` tokens is being added.
