# CS238/AA238 Final Project

## Getting Started

1. Install required system packages.

```bash
sudo apt-get install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
```

2. Install [conda](https://docs.conda.io/en/latest/) if you don't have it on your machine, ideally by installing [Anaconda](https://www.anaconda.com/).

3. Create the conda environment for the project.

```bash
conda env create -f env.yml
```

4. Activate the environment.

```bash
conda activate pinball
```

5. Install pre-commit hooks while in the root of this project.

```bash
pre-commit install
```

6. (Optional) Start the Jupyter notebook server. Make sure notebooks are saved in `project/notebooks`.

```bash
make start_jupyter
```
