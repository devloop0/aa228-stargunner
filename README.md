# CS238/AA238 Final Project

## Getting Started

1. Install [conda](https://docs.conda.io/en/latest/) if you don't have it on your machine, ideally by installing [Anaconda](https://www.anaconda.com/).

2. Create the conda environment for the project.

```bash
conda env create -f env.yml
```

3. Activate the environment.

```bash
conda activate pinball
```

4. Install pre-commit hooks while in the root of this project.

```bash
pre-commit install
```

5. (Optional) Start the Jupyter notebook server. Make sure notebooks are saved in `project/notebooks`.

```bash
make start_jupyter
```
