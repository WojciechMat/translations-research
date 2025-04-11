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
