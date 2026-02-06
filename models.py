import jax
import haiku as hk
import jax.numpy as jnp
from jax.example_libraries import optimizers
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import neural_tangents as nt
import functools
import operator
import optax
import copy
import sys
from utils import bind, _sub, _add
# 假设您有 modified_resnets 文件，如果没有请注释掉或替换为标准 ResNet
import modified_resnets


# =============================================================================
# NTK Linearization Logic (保留原样)
# =============================================================================
@functools.partial(jax.jit, static_argnums=(3, 6, 7, 8))
def linear_forward(params, params2, state, net_fn, rng, images, is_training=False, centering=False,
                   return_components=False):
    dparams = _sub(params2, params)
    f_0, df_0, state = jax.jvp(lambda param: net_fn(param, state, rng, images, is_training=is_training), (params,),
                               (dparams,), has_aux=True)
    if return_components:
        if centering:
            return df_0, {'state': state, 'f': f_0, 'df': df_0}
        return _add(f_0, df_0), {'state': state, 'f': f_0, 'df': df_0}

    if centering:
        return df_0, state
    return _add(f_0, df_0), state


# =============================================================================
# Existing Models (保留 ResNet 和 MLP)
# =============================================================================

def get_resnet(n_classes):
    def _forward_resnet18(x, is_training):
        # use a 3x3 kernel size with stride 1 for the first layer because we are using 32x32 images
        net = modified_resnets.ResNet18(n_classes, initial_conv_config={'kernel_shape': 3, 'stride': 1})
        return net(x, is_training)

    net_forward = hk.transform_with_state(_forward_resnet18)
    return net_forward.init, net_forward.apply


def _forward_narrow_mlp(x, is_training):
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(256), jax.nn.relu,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)


def _forward_wide_mlp(x, is_training):
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(2048), jax.nn.relu,
        hk.Linear(2048), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)


def get_narrow_mlp():
    net_forward = hk.transform_with_state(_forward_narrow_mlp)
    return net_forward.init, net_forward.apply


def get_wide_mlp():
    net_forward = hk.transform_with_state(_forward_wide_mlp)
    return net_forward.init, net_forward.apply


# =============================================================================
# New Models: LeNet & CifarVGG
# =============================================================================

def get_lenet(n_classes):
    """
    LeNet-5 变体。
    专为 MNIST / Fashion-MNIST 设计。
    结构: Conv -> Pool -> Conv -> Pool -> FC -> FC -> FC
    """

    def _forward_lenet(images, is_training=True):
        # Haiku 会自动处理通道数 (MNIST 为 1)
        net = hk.Sequential([
            # Layer 1
            hk.Conv2D(output_channels=6, kernel_shape=5, padding='SAME'),
            jax.nn.relu,
            hk.MaxPool(window_shape=2, strides=2, padding='VALID'),

            # Layer 2
            hk.Conv2D(output_channels=16, kernel_shape=5, padding='VALID'),
            jax.nn.relu,
            hk.MaxPool(window_shape=2, strides=2, padding='VALID'),

            # Layer 3 (Fully Connected)
            hk.Flatten(),
            hk.Linear(120),
            jax.nn.relu,
            hk.Linear(84),
            jax.nn.relu,

            # Output
            hk.Linear(n_classes)
        ])
        return net(images)

    net_forward = hk.transform_with_state(_forward_lenet)
    return net_forward.init, net_forward.apply


class CifarVGG(hk.Module):
    """
    VGG-style ConvNet (Small VGG)。
    专为 CIFAR-10 / CIFAR-100 设计。
    比简单的 CNN 深，带有 Batch Norm，比 ResNet18 轻量。
    """

    def __init__(self, n_classes, name=None):
        super().__init__(name=name)
        self.n_classes = n_classes

    def __call__(self, x, is_training=True):
        # 定义一个 Conv-BN-ReLU 块
        def conv_block(x, channels):
            x = hk.Conv2D(output_channels=channels, kernel_shape=3, padding='SAME', with_bias=False)(x)
            x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training)
            x = jax.nn.relu(x)
            return x

        # --- Block 1 ---
        x = conv_block(x, 32)
        x = conv_block(x, 32)
        x = hk.MaxPool(window_shape=2, strides=2, padding='VALID')(x)
        # Dropout 可选，这里为了对抗训练稳定性暂时不加，或者设得很低

        # --- Block 2 ---
        x = conv_block(x, 64)
        x = conv_block(x, 64)
        x = hk.MaxPool(window_shape=2, strides=2, padding='VALID')(x)

        # --- Block 3 ---
        x = conv_block(x, 128)
        x = conv_block(x, 128)
        x = hk.MaxPool(window_shape=2, strides=2, padding='VALID')(x)

        # --- Classifier ---
        x = hk.Flatten()(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        # x = hk.Dropout(0.5)(x, rng=hk.next_rng_key() if is_training else None) # 如果需要 dropout
        x = hk.Linear(self.n_classes)(x)

        return x


def get_cifar_cnn(n_classes):
    def _forward_cifar_cnn(images, is_training=True):
        net = CifarVGG(n_classes)
        return net(images, is_training)

    net_forward = hk.transform_with_state(_forward_cifar_cnn)
    return net_forward.init, net_forward.apply


# =============================================================================
# Model Dispatcher
# =============================================================================

def get_model(model_name, n_classes):
    if model_name == 'resnet18':
        return get_resnet(n_classes)
    elif model_name == 'mlp_skinny':
        return get_narrow_mlp()
    elif model_name == 'mlp_wide':
        return get_wide_mlp()
    elif model_name == 'lenet':  # 新增: 适合 MNIST/FMNIST
        return get_lenet(n_classes)
    elif model_name == 'cifar_cnn':  # 新增: 适合 CIFAR10/100
        return get_cifar_cnn(n_classes)
    else:
        raise ValueError("Invalid model: {}".format(model_name))