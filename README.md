# Jet Classification using Classical and Neural Models

This repository explores jet classification in high-energy physics using both classical machine learning and deep learning models. It leverages the [JetNet](https://github.com/jet-net/JetNet/tree/main) dataset to distinguish between gluon, top quark, and W boson jets based on particle and jet-level features.

## Contents

### ✅ `logistic_reg.py`
- Implements **logistic regression** on flattened particle data using `scikit-learn`.
- Applies **one-hot encoding** on jet types and preprocesses data using JetNet utilities.
- Reports classification accuracy on the validation set.
- Contains an in-progress transformer-based architecture for future experimentation.

### ✅ `mlp.py`
- Implements a simple **Multi-Layer Perceptron (MLP)** in PyTorch.
- Uses the same data pipeline as `logistic_reg.py`.
- Trains a two-layer MLP to predict the jet class and evaluates its performance.

### ✅ `DeepsetModel.ipynb`
- Implements a **Deep Sets architecture** for variable-sized particle inputs.
- Trains a permutation-invariant model to classify jets using particle features.
- Contains model training, evaluation, and insight into model structure.

## Dataset

- The dataset is handled via the [JetNet API](https://github.com/jet-net/JetNet/tree/main) and is automatically downloaded.
- Features used:
  - **Particle features**: `etarel`, `phirel`, `ptrel`, `mask`
  - **Jet features**: `type`, `pt`, `eta`, `mass`

## Models Compared

| Model              | Framework     | Description                                 |
|--------------------|---------------|---------------------------------------------|
| Logistic Regression| scikit-learn  | Baseline model with flattened particle data |
| MLP                | PyTorch       | Shallow neural network classifier           |
| Deep Sets          | PyTorch       | Permutation-invariant neural architecture   |

## Installation

```bash
pip install torch scikit-learn jetnet


@inproceedings{Kansal_MPGAN_2021,
  author = {Kansal, Raghav and Duarte, Javier and Su, Hao and Orzari, Breno and Tomei, Thiago and Pierini, Maurizio and Touranakou, Mary and Vlimant, Jean-Roch and Gunopulos, Dimitrios},
  booktitle = "{Advances in Neural Information Processing Systems}",
  editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
  pages = {23858--23871},
  publisher = {Curran Associates, Inc.},
  title = {Particle Cloud Generation with Message Passing Generative Adversarial Networks},
  url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf},
  volume = {34},
  year = {2021},
  eprint = {2106.11535},
  archivePrefix = {arXiv},
}

@article{Kansal_JetNet_2023,
  author = {Kansal, Raghav and Pareja, Carlos and Hao, Zichun and Duarte, Javier},
  doi = {10.21105/joss.05789},
  journal = {Journal of Open Source Software},
  number = {90},
  pages = {5789},
  title = {{JetNet: A Python package for accessing open datasets and benchmarking machine learning methods in high energy physics}},
  url = {https://joss.theoj.org/papers/10.21105/joss.05789},
  volume = {8},
  year = {2023}
}
