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
```
## 📂 File Structure

* `run_exp.py`: Main script for **Ours** (Stage 1 & Stage 2) and **FedAvg-AT**.
* `run_fed_mart.py`: Implementation of **Fed-MART**.
* `run_fed_awp.py`: Implementation of **Fed-AWP**.
* `run_fed_calfat.py`: Implementation of **CalFAT**.
* `models.py`: Network architectures (LeNet, CifarCNN, ResNet18).
* `test_functions.py`: Adversarial attack implementations (PGD, FGSM, MIM, C&W).
* `data.py`: Data loading and Non-IID partitioning logic.

## 🚀 Usage: Running Our Method

Our method consists of two stages. Stage 1 trains a benign anchor model, and Stage 2 performs linearized adversarial fine-tuning.

### 1. Stage 1: Standard Benign Pre-training
Train a standard model on clean data to serve as the linearization anchor ($w_0$).

```bash
# Example: FMNIST using LeNet (100 Epochs Benign)
python run_exp.py \
    --dataset fmnist \
    --model lenet \
    --standard_epochs 101 \
    --linear_epochs 0 \
    --loaders CC \
    --attack_method pgd \
    --save_path sgd_benign100 \
    --constant_save \
    --random_seed 0 \
    --skip_second_test \
    --is_iid True
```
### 2. Stage 2: Linearized Adversarial Fine-tuning
Load the checkpoint from Stage 1 (e.g., epoch 50) and perform linearized adversarial training.

```bash
# Load checkpoint from 'sgd_benign100/phase1_epoch_50.pkl'
# Note: The script automatically handles path prefixes.
python run_exp.py \
    --base_model_path sgd_benign100/phase1_epoch_50.pkl \
    --dataset fmnist \
    --model lenet \
    --standard_epochs 0 \
    --linear_epochs 51 \
    --loaders CA \
    --attack_method pgd \
    --save_path sgd_benign50_to_linear_adv50 \
    --random_seed 0 \
    --skip_first_test \
    --constant_save_linear \
    --is_iid True
```
## 📊 Usage: Running Baselines

We provide implementations for state-of-the-art federated adversarial training methods.

### 1. Fed-MART
Federated version of Misclassification Aware Regularized Training.

```bash
python run_fed_mart.py \
    --dataset cifar10 \
    --model cifar_cnn \
    --attack_method fgsm \
    --save_path mart_adv100 \
    --is_iid True
```
### 2. Fed-AWP
Federated Adversarial Weight Perturbation (SOTA method).

```bash
python run_fed_awp.py \
    --dataset cifar10 \
    --model cifar_cnn \
    --attack_method pgd \
    --save_path awp_adv100 \
    --is_iid True
```
### 3. CalFAT
Calibrated Federated Adversarial Training for Non-IID label skew.

```bash
python run_fed_calfat.py \
    --dataset cifar10 \
    --model cifar_cnn \
    --attack_method pgd \
    --save_path calfat_adv100 \
    --is_iid True
```
### 4. FedAvg-AT (Naive Adversarial Training)
Standard Federated Averaging with local PGD training.

```bash
python run_exp.py \
    --dataset cifar10 \
    --model cifar_cnn \
    --standard_epochs 101 \
    --linear_epochs 0 \
    --loaders AC \
    --attack_method pgd \
    --save_path sgd_adv100 \
    --constant_save \
    --random_seed 0 \
    --skip_second_test \
    --is_iid True
```
## ⚙️ Arguments

Key arguments explanation for `run_exp.py`:

* `--dataset`: `cifar10`, `fmnist`, or `cifar100`.
* `--model`: `lenet` (for FMNIST), `cifar_cnn` (VGG-Small), or `resnet18`.
* `--is_iid`: `True` for IID data, `False` for Non-IID (Dirichlet).
* `--attack_method`: `pgd`, `fgsm`, `mim`, or `cw`.
* `--loaders`: Training mode configuration string.
    * `CC`: Phase 1 Clean, Phase 2 Clean.
    * `CA`: Phase 1 Clean, Phase 2 Adv (**Ours**).
    * `AC`: Phase 1 Adv, Phase 2 Clean (**FedAvg-AT**).
* `--eps`: Adversarial perturbation budget (e.g., 8.0, code scales by 1/255 automatically).
* `--base_model_path`: Path to the pickle checkpoint for fine-tuning.
* `--save_path`: Directory to save logs and checkpoints.

## 📝 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{2026FedLAT,
  title={Enhancing Adversarial Robustness in Federated Learning via NTK-Based Linearized Training},
  author={Ping Luo},
  journal={},
  year={2026}
}
