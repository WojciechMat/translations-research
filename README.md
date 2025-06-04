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
The "brutal" translator is a simple word-for-word translation system that uses a dictionary to translate each word independently. If there is no single-word translation the word is kept the same as original.
```sh
# Run with default settings (uses a test dictionary and the first 10 examples)
python -m translations.main

# Override configuration values on the command line
python -m translations.main slice_spec=":20" data.random_seed=123

# Use a different config file
python -m translations.main --config-name=moses

# Build a dictionary from gemini
python translations.main dictionary.build=true dictionary.source=gemini
```
### Moses Translation
Described in `docs/moses.md`
### Using Google Sheets for experiment tracking
Follow instructions on `docs/using_sheets.md` to use google sheets for experiment tracking.
### Results overwiev
Once you have some data in the google sheets, you can run results overview
```sh
python -m streamlit translations/results_dashboard.py
```
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
