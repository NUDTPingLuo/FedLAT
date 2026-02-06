# Enhancing Adversarial Robustness in Federated Learning via NTK-Based Linearized Training

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX%20%7C%20Haiku-blue)](https://github.com/google/jax)
[![Python: 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)

This repository contains the official JAX/Haiku implementation for the paper **"Enhancing Adversarial Robustness in Federated Learning via NTK-Based Linearized Training"**.

We propose a novel two-stage Federated Adversarial Training framework (**Fed-LAT**). By leveraging the **Neural Tangent Kernel (NTK)** linearization dynamics in the fine-tuning stage, our method achieves superior adversarial robustness in **Non-IID** federated settings while maintaining high computational efficiency compared to methods like Fed-AWP.

## 📋 Table of Contents
- [Requirements & Installation](#-requirements--installation)
- [File Structure](#-file-structure)
- [Usage: Running Our Method](#-usage-running-our-method)
    - [Stage 1: Standard Benign Pre-training](#1-stage-1-standard-benign-pre-training)
    - [Stage 2: Linearized Adversarial Fine-tuning](#2-stage-2-linearized-adversarial-fine-tuning)
- [Usage: Running Baselines](#-usage-running-baselines)
    - [Fed-MART](#1-fed-mart)
    - [Fed-AWP](#2-fed-awp)
    - [CalFAT](#3-calfat)
    - [FedAvg-AT](#4-fedavg-at-naive-adversarial-training)
- [Arguments](#-arguments)

## 🛠 Requirements & Installation

**System Tested:** Ubuntu 22.04, Python 3.10, Torch 2.1.2, CUDA 11.

To ensure reproducibility, please follow the specific installation steps below:

```bash
# 1. Create and activate conda environment
conda create -n fed_ntk python=3.10
conda activate fed_ntk

# 2. Install basic dependencies
pip install torch==2.1.2 torchvision

# 3. Install Neural Tangents
pip install neural-tangents==0.6.0

# 4. Clean up any pre-existing JAX versions to avoid conflicts
pip uninstall jax jaxlib -y

# 5. Install JAX (Specific CUDA 11 wheel)
# Note: Ensure this matches your CUDA version. For CUDA 12, check JAX official releases.
pip install [https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl](https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl)
pip install jax==0.4.13

# 6. Install Haiku
pip install dm-haiku==0.0.9

# 7. Install other utilities (Note the --no-deps flags to prevent version conflicts)
pip install optax==0.1.7 --no-deps
pip install scipy==1.12.0
pip install chex==0.1.7 --no-deps
pip install tf2jax==0.3.0 --no-deps

# Verify installation
python -c "import jax; print(f'JAX Devices: {jax.devices()}')"
