# English-Polish Translation Project

This project implements various translation approaches for English to Polish translation, starting with a simple word-for-word "brutal" translation approach.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/en-pl-translation.git
cd en-pl-translation

# Install dependencies
pip install -r requirements.txt
```
## Usage
### Brutal Translation
The "brutal" translator is a simple word-for-word translation system that uses a dictionary to translate each word independently.
```sh
# Run with default settings (uses a test dictionary and the first 10 examples)
python main.py

# Override configuration values on the command line
python main.py slice_spec=":20" data.random_seed=123

# Use a different config file
python main.py --config-name=smt  # Future implementation

# Build a dictionary from OPUS
python main.py dictionary.build=true dictionary.source=opus

# Build a dictionary from diki
python main.py dictionary.build=true dictionary.source=diki
```
## Configuration Options
The project uses Hydra for configuration management. The main configuration options are:

- `data.source_lang`: Source language code (default: "en")
- `data.target_lang`: Target language code (default: "pl")
- `data.dataset_path`: Path to the dataset (default: "hf_datasets/EuroparlEnPl")
- `slice_spec`: Slice specification (e.g., ":50", "-50:", "10:20") (default: ":10")
- `dictionary.path`: Path to the dictionary file (default: "dictionaries/en_pl_dictionary.json")
- `dictionary.keep_unknown`: Whether to keep unknown words (default: true)
- `dictionary.lowercase`: Whether to lowercase input text (default: true)
- `dictionary.build`: Whether to build a dictionary (default: false)
- `dictionary.source`: Source for building the dictionary ("diki" or "opus") (default: "opus")
# Development

### Code Style

This repository uses pre-commit hooks with forced Python formatting ([black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit`, the files that were altered or added will be checked and corrected. Tools such as `black` and `isort` may modify files locally—in which case you must `git add` them again. You might also be prompted to make some manual fixes.

To run the hooks against all files without running a commit:

```sh
pre-commit run --all-files
```
