# README

## How to build documentation (need `sphinx`)

```shell
sphinx-build -M html docs/source docs/build
```

or

```shell
cd docs
make html
```

The documentation including installation manual can then be accessed from `docs/build/html/index.html`

## Project structure

- `manybodytensor`: Main module
- `docs`: Documentation
- `tests`: Functional tests
- `scripts`: Benchmark scripts
- `notebooks`: Benchmark Jupyter Notebooks
