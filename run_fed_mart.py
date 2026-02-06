import jax
import haiku as hk
import jax.numpy as jnp
from jax.example_libraries import optimizers
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import functools
import operator
import optax
import copy
import models
import pickle
import os
import argparse
import random
from collections import defaultdict
import csv
import numpy.linalg as la

# 导入必要的工具函数
from test_functions import perturb, test
import data


# =============================================================================
# MART Loss 定义 (核心逻辑)
# =============================================================================

def mart_loss_calculation(logits_clean, logits_adv, labels, beta):
    """
    计算 MART 损失:
    L = BCE(f(x_adv), y) + beta * KL(f(x_clean) || f(x_adv)) * (1 - p_y(x_clean))
    """
    num_classes = logits_clean.shape[-1]
    labels_oh = jax.nn.one_hot(labels, num_classes)

    # --- 1. Boosted Cross Entropy (BCE) on Adversarial Examples ---
    probs_adv = jax.nn.softmax(logits_adv)

    # p_y: 真实类别的预测概率
    p_adv_y = jnp.sum(probs_adv * labels_oh, axis=-1)

    # p_not_y_max: 非真实类别中的最大预测概率
    p_adv_not_y = probs_adv * (1.0 - labels_oh)
    max_p_adv_not_y = jnp.max(p_adv_not_y, axis=-1)

    # BCE Loss = -log(p_y) - log(1 - max(p_not_y))
    eps = 1e-8
    loss_bce = -jnp.log(p_adv_y + eps) - jnp.log(1.0 - max_p_adv_not_y + eps)

    # --- 2. KL Divergence (Clean || Adv) ---
    probs_clean = jax.nn.softmax(logits_clean)
    log_probs_clean = jax.nn.log_softmax(logits_clean)
    log_probs_adv = jax.nn.log_softmax(logits_adv)

    # KL(P||Q) = sum(P * (logP - logQ))
    kl_div = jnp.sum(probs_clean * (log_probs_clean - log_probs_adv), axis=-1)

    # --- 3. Conditional Weighting ---
    p_clean_y = jnp.sum(probs_clean * labels_oh, axis=-1)
    weight = 1.0 - p_clean_y

    # --- Final Loss ---
    total_loss = jnp.mean(loss_bce + beta * kl_div * weight)

    return total_loss


# [修正] 移除了 @jax.jit，因为该函数会在 do_mart_training_step 内部被调用
# 这样可以避免参数传递时的静态/动态冲突
def mart_loss_fn(params, state, net_fn, rng, clean_images, adv_images, labels, beta, is_training=True):
    """
    MART 训练的 Loss 函数封装
    """
    # 1. 前向传播：Clean Images
    logits_clean, state = net_fn(params, state, rng, clean_images, is_training=is_training)

    # 2. 前向传播：Adv Images
    logits_adv, state = net_fn(params, state, rng, adv_images, is_training=is_training)

    # 3. 计算 Loss
    loss = mart_loss_calculation(logits_clean, logits_adv, labels, beta)

    # 计算准确率 (基于对抗样本)
    acc = jnp.mean(logits_adv.argmax(1) == labels)

    return loss, {'net_state': state, 'acc': acc}


# =============================================================================
# Training Steps
# =============================================================================

# [修正] static_argnums=(2, 4, 10)
# Index 2: net_fn (函数，静态)
# Index 4: optimizer_update (函数，静态)
# Index 10: is_training (布尔值，静态)
# Index 3 (opt_state) 现在是动态的，修复了 "Non-hashable static argument" 错误
@functools.partial(jax.jit, static_argnums=(2, 4, 10))
def do_mart_training_step(params, net_state, net_fn, opt_state, optimizer_update, rng,
                          clean_images, adv_images, labels, beta, is_training=True):
    """
    执行一步 MART 训练参数更新
    """
    # 计算梯度
    [loss, lf_dict], grads = jax.value_and_grad(mart_loss_fn, has_aux=True)(
        params, net_state, net_fn, rng, clean_images, adv_images, labels, beta, is_training
    )

    net_state = lf_dict['net_state']
    acc = lf_dict['acc']

    # 应用梯度更新
    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, net_state, opt_state, acc


# =============================================================================
# 数据处理与辅助工具
# =============================================================================

def get_client_indices(labels_np, num_clients, iid=True, alpha=0.1):
    dataset_len = len(labels_np)
    indices_list = []
    if iid:
        indices = np.random.permutation(dataset_len)
        sizes = [dataset_len // num_clients] * num_clients
        sizes[-1] += dataset_len % num_clients
        start = 0
        for size in sizes:
            indices_list.append(indices[start:start + size])
            start += size
    else:
        num_classes = len(np.unique(labels_np))
        label_indices = [np.where(labels_np == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            np.random.shuffle(label_indices[c])
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(label_indices[c])).astype(int)[:-1]
            split = np.split(label_indices[c], split_points)
            for i, part in enumerate(split):
                client_indices[i].extend(part)
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
            indices_list.append(np.array(client_indices[i]))
    return indices_list


def fed_avg(param_list, local_size_list):
    weights = np.array(local_size_list)
    weights = weights / np.sum(weights)
    avg_params = copy.deepcopy(param_list[0])
    for k in avg_params.keys():
        avg_params[k] = jax.tree_util.tree_map(
            lambda *params: sum(w * p for w, p in zip(weights, params)),
            *[client_params[k] for client_params in param_list]
        )
    return avg_params


# =============================================================================
# 主程序
# =============================================================================

def main():
    print("JAX Version:", jax.__version__)
    print("JAX Devices:", jax.devices())

    # --- 参数设置 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: cifar10/fmnist')
    parser.add_argument('--model', type=str, default='resnet18', help='model arch')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')

    # 攻击相关参数
    parser.add_argument('--attack_method', type=str, default='pgd', choices=['fgsm', 'pgd', 'mim', 'cw'])
    parser.add_argument('--eps', type=float, default=8.00, help='Perturbation budget (scaled by 1/255 inside)')
    parser.add_argument('--mart_beta', type=float, default=6.0, help='MART hyperparameter beta (weight for KL)')

    # 联邦学习与保存相关
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='Result save path')
    parser.add_argument('--constant_save', action='store_true', help='Save model checkpoints frequently')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--base_model_path', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--is_iid', type=bool, default=True)

    args = parser.parse_args()

    # --- GPU 数据加载 ---
    print(">>> Loading Data to GPU...")
    train_images_np, train_labels_np = data.get_data_and_labels(args.dataset)

    # 维度修正 (NCHW -> NHWC)
    if train_images_np.ndim == 4 and train_images_np.shape[1] in [1, 3]:
        if train_images_np.shape[2] > train_images_np.shape[1]:
            print(f"Transposing data from {train_images_np.shape} to NHWC format.")
            train_images_np = np.transpose(train_images_np, (0, 2, 3, 1))

    all_train_images = jnp.array(train_images_np)
    all_train_labels = jnp.array(train_labels_np)

    # Non-IID 设置
    num_clients = 10
    alpha_dirichlet = 0.1
    distribute = 'IID' if args.is_iid else 'Non_IID'
    client_indices_list = get_client_indices(train_labels_np, num_clients, iid=args.is_iid, alpha=alpha_dirichlet)

    # 测试集加载
    test_loader = data.get_loader(args.dataset, train=False, batch_size=100, shuffle=False)

    # 保存路径
    args.save_path = f'results_mart/{args.dataset}/{args.attack_method}_{distribute}_{args.save_path}'
    os.makedirs(args.save_path, exist_ok=True)

    rng = jax.random.PRNGKey(args.random_seed)
    print(f"Random Seed: {args.random_seed}, Attack: {args.attack_method}, MART Beta: {args.mart_beta}")

    # --- 模型初始化 ---
    net_forward_init, net_forward_apply = models.get_model(args.model, data.get_n_classes(args.dataset))

    # Dummy init
    dummy_input = all_train_images[0:1]
    params, net_state = net_forward_init(rng, dummy_input, is_training=True)

    # 优化器
    optimizer_init, optimizer_update = optax.chain(optax.sgd(args.lr, momentum=0.9))
    opt_state = optimizer_init(params)

    # 断点续训加载
    if len(args.base_model_path) > 0:
        print(f'Loading from saved model: {args.base_model_path}')
        with open(args.base_model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        net_state = checkpoint['net_state']

    # CSV 记录
    csv_file_path = os.path.join(args.save_path, f'result_pure_mart.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'clean', 'adv_acc'])

    # =========================================================================
    # Pure Fed-MART Training Loop
    # =========================================================================
    print(">>> Starting Pure Fed-MART Training")
    BATCH_SIZE = 128

    # 攻击参数
    adv_eps_val = args.eps / 255.0
    adv_iters = 10

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        client_params_list = []
        client_net_state_list = []
        local_size_list = []

        # 遍历客户端
        for client_id in range(num_clients):
            # 1. 客户端下载全局模型
            local_params = copy.deepcopy(params)
            local_net_state = copy.deepcopy(net_state)
            local_opt_state = optimizer_init(local_params)

            # 2. 获取本地数据
            indices = client_indices_list[client_id]
            num_samples = len(indices)
            num_batches = num_samples // BATCH_SIZE
            indices_jax = jnp.array(indices)

            rng, shuffle_key = jax.random.split(rng)
            shuffled_indices = jax.random.permutation(shuffle_key, indices_jax, independent=True)

            # 3. 本地训练循环
            for b in range(num_batches):
                batch_idx = shuffled_indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
                batch_images_clean = all_train_images[batch_idx]
                batch_labels = all_train_labels[batch_idx]

                rng, step_key = jax.random.split(rng)

                # A. 生成对抗样本 (使用当前 local_params)
                batch_images_adv = perturb(
                    local_params, local_params, local_net_state, net_forward_apply, step_key,
                    batch_images_clean, batch_labels,
                    adv_eps_val, 2 * adv_eps_val / adv_iters, adv_iters,
                    linear=False, centering=False, attack_method=args.attack_method,
                    is_training=True
                )

                # B. 计算 MART Loss 并更新参数
                loss, local_params, local_net_state, local_opt_state, acc = do_mart_training_step(
                    local_params, local_net_state, net_forward_apply,
                    local_opt_state, optimizer_update, step_key,
                    batch_images_clean, batch_images_adv, batch_labels,
                    args.mart_beta, is_training=True
                )

            # 收集本地更新结果
            client_params_list.append(local_params)
            client_net_state_list.append(local_net_state)
            local_size_list.append(num_samples)

        # 4. 服务器聚合 (FedAvg)
        params = fed_avg(client_params_list, local_size_list)
        net_state = fed_avg(client_net_state_list, local_size_list)

        # 5. 评估与保存
        should_eval_adv = (epoch % 5 == 0) or (epoch == args.epochs - 1)

        # 为了兼容 test 函数签名，lin_params 传入 params
        clean_acc, adv_acc = test(
            params, params, net_state, net_forward_apply, rng, test_loader,
            linear=False, make_adv_examples=should_eval_adv,
            attack_method=args.attack_method, adv_eps=args.eps, short=True
        )
        if not should_eval_adv: adv_acc = 0.0

        print(f"Ep {epoch} | Clean: {clean_acc:.2%} | Adv (MART): {adv_acc:.2%}")

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, clean_acc, adv_acc])

        if args.constant_save and (epoch % 10 == 0 or epoch == args.epochs - 1):
            pickle.dump({'params': params, 'net_state': net_state},
                        open(f'./{args.save_path}/epoch_{epoch}.pkl', 'wb'))


if __name__ == '__main__':
    main()