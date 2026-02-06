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
from jax import lax

# 导入基础测试函数 (test 用于评估)
from test_functions import test
import data


# =============================================================================
# 1. 核心: CalFAT Loss (Logit Adjustment)
# =============================================================================

def calfat_loss_calculation(logits, labels, class_priors, tau=1.0):
    """
    计算经过 Logit Adjustment 校准的 Cross Entropy Loss
    Args:
        logits: 模型输出 [B, num_classes]
        labels: 真实标签 [B]
        class_priors: 本地数据的类别分布频率 [num_classes]
        tau: 校准温度系数 (通常 1.0)
    """
    # 避免 log(0)
    log_priors = jnp.log(class_priors + 1e-12)

    # --- 核心操作: Logit Adjustment ---
    # 减去 log(p) 是为了消除本地数据不平衡带来的 Bias
    # 让模型学习到的 Logits 是相对于“全局均匀分布”的
    logits_calibrated = logits - tau * log_priors

    # 标准 CE Loss
    labels_oh = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits_calibrated, labels_oh).mean()

    return loss


# [修正] 移除 @jax.jit，避免静态参数配置错误，反正它在外层 JIT 环境中运行
def calfat_loss_fn(params, state, net_fn, rng, images, labels, class_priors, tau=1.0, is_training=True):
    """
    封装后的 CalFAT Loss 函数
    """
    logits, state = net_fn(params, state, rng, images, is_training=is_training)
    loss = calfat_loss_calculation(logits, labels, class_priors, tau)

    # 准确率仍基于原始 logits 计算 (Evaluation 时不需要校准)
    acc = jnp.mean(logits.argmax(1) == labels)

    return loss, {'net_state': state, 'acc': acc}


# =============================================================================
# 2. 核心: Calibrated Perturbation (对抗样本生成)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(2))
def clamp_by_norm(x, r, norm='l_inf'):
    if norm == 'l_inf':
        return jnp.clip(x, -r, r)
    return x


# 单步攻击 (Calibrated)
# 注意：这里不需要 JIT，因为它被 perturb_calfat 调用，而 perturb_calfat 已经 JIT 了
def do_calibrated_perturbation_step(params, net_state, net_fn, rng, images0, images, labels,
                                    class_priors, tau, eps, alpha, is_training=True):
    # 对 calfat_loss_fn 求梯度 (针对 images)
    # argnums=4 对应 images
    grads = jax.grad(
        lambda i, l: calfat_loss_fn(params, net_state, net_fn, rng, i, l, class_priors, tau, is_training)[0])(images,
                                                                                                              labels)

    grads = jnp.sign(grads)
    images = images + alpha * grads
    images = jnp.clip(images, 0., 1.)

    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm='l_inf')
    images = images0 + d_images
    return images


# [修正] static_argnums=(2, 10, 11, 12)
# 2: net_fn (函数)
# 10: iters (整数)
# 11: attack_method (字符串)
# 12: is_training (布尔)
@functools.partial(jax.jit, static_argnums=(2, 10, 11, 12))
def perturb_calfat(params, net_state, net_fn, rng, images, labels,
                   class_priors, tau, eps, alpha, iters,
                   attack_method='pgd', is_training=True):
    """
    生成基于 CalFAT Loss 的对抗样本
    """
    images0 = images

    if attack_method == 'pgd' or attack_method == 'mim':  # 统一处理
        # Random Init
        noise = jax.random.uniform(rng, images.shape, minval=-eps, maxval=eps)
        images = images + noise
        images = jnp.clip(images, 0., 1.)
        d_images = clamp_by_norm(images - images0, eps, norm='l_inf')
        images = images0 + d_images

        def body_fun(carry, _):
            curr_images = carry
            next_images = do_calibrated_perturbation_step(
                params, net_state, net_fn, rng, images0, curr_images, labels,
                class_priors, tau, eps, alpha, is_training
            )
            return next_images, None

        final_images, _ = lax.scan(body_fun, images, None, length=iters)
        return final_images

    elif attack_method == 'fgsm':
        return do_calibrated_perturbation_step(
            params, net_state, net_fn, rng, images0, images, labels,
            class_priors, tau, eps, eps, is_training
        )

    return images


# =============================================================================
# 3. Training Step
# =============================================================================

# [修正] static_argnums=(2, 4, 10) -> net_fn, optimizer_update, is_training
@functools.partial(jax.jit, static_argnums=(2, 4, 10))
def do_calfat_training_step(params, net_state, net_fn, opt_state, optimizer_update, rng,
                            adv_images, labels, class_priors, tau, is_training=True):
    # 使用校准 Loss 计算梯度并更新
    [loss, lf_dict], grads = jax.value_and_grad(calfat_loss_fn, has_aux=True)(
        params, net_state, net_fn, rng, adv_images, labels, class_priors, tau, is_training
    )

    net_state = lf_dict['net_state']
    acc = lf_dict['acc']

    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, net_state, opt_state, acc


# =============================================================================
# 4. 辅助工具
# =============================================================================

def get_class_counts(labels_np, num_classes):
    """计算类别频率 priors"""
    counts = np.zeros(num_classes)
    unique, counts_idx = np.unique(labels_np, return_counts=True)
    counts[unique] = counts_idx
    # 归一化为概率，加平滑避免 0
    priors = (counts + 1e-12) / (np.sum(counts) + 1e-12 * num_classes)
    return priors


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='model arch')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')

    parser.add_argument('--attack_method', type=str, default='pgd', choices=['fgsm', 'pgd', 'cw'])
    parser.add_argument('--eps', type=float, default=8.00, help='Perturbation budget (8/255)')

    # CalFAT 参数
    parser.add_argument('--cal_tau', type=float, default=1.0, help='Calibration temperature (Logit Adjustment)')

    parser.add_argument('--save_path', type=str, default='./saved_models/', help='Result save path')
    parser.add_argument('--constant_save', action='store_true', help='Save model checkpoints')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--base_model_path', type=str, default='')
    parser.add_argument('--is_iid', type=bool, default=True)

    args = parser.parse_args()

    # --- Data Loading ---
    print(">>> Loading Data to GPU...")
    train_images_np, train_labels_np = data.get_data_and_labels(args.dataset)
    num_classes = data.get_n_classes(args.dataset)

    # [修正] 兼容 1通道和3通道数据的转置逻辑
    if train_images_np.ndim == 4 and train_images_np.shape[1] in [1, 3]:
        if train_images_np.shape[2] > train_images_np.shape[1]:
            print(f"Transposing data from {train_images_np.shape} to NHWC format.")
            train_images_np = np.transpose(train_images_np, (0, 2, 3, 1))

    all_train_images = jnp.array(train_images_np)
    all_train_labels = jnp.array(train_labels_np)

    # Non-IID Setup
    num_clients = 10
    alpha_dirichlet = 0.1  # 极度 Non-IID，测试 CalFAT 的最佳场景
    distribute = 'IID' if args.is_iid else 'Non_IID'
    client_indices_list = get_client_indices(train_labels_np, num_clients, iid=is_iid, alpha=alpha_dirichlet)

    # 预计算每个客户端的 Class Priors (用于校准)
    print(">>> Computing Client Class Priors...")
    client_priors_list = []
    for i in range(num_clients):
        local_labels = train_labels_np[client_indices_list[i]]
        priors = get_class_counts(local_labels, num_classes)
        client_priors_list.append(jnp.array(priors))

    test_loader = data.get_loader(args.dataset, train=False, batch_size=100, shuffle=False)

    args.save_path = f'results_calfat/{args.dataset}/{args.attack_method}_{distribute}_{args.save_path}'
    os.makedirs(args.save_path, exist_ok=True)

    rng = jax.random.PRNGKey(args.random_seed)

    # --- Model Init ---
    net_forward_init, net_forward_apply = models.get_model(args.model, num_classes)
    dummy_input = all_train_images[0:1]
    params, net_state = net_forward_init(rng, dummy_input, is_training=True)

    optimizer_init, optimizer_update = optax.chain(optax.sgd(args.lr, momentum=0.9))
    opt_state = optimizer_init(params)

    if len(args.base_model_path) > 0:
        print(f'Loading checkpoint: {args.base_model_path}')
        with open(args.base_model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        net_state = checkpoint['net_state']

    csv_file_path = os.path.join(args.save_path, f'result_calfat.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'clean', 'adv_acc'])

    # =========================================================================
    # CalFAT Training Loop
    # =========================================================================
    print(f">>> Starting CalFAT Training (Tau={args.cal_tau})")
    BATCH_SIZE = 128
    adv_eps_val = args.eps / 255.0
    adv_iters = 10

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        client_params_list = []
        client_net_state_list = []
        local_size_list = []

        for client_id in range(num_clients):
            local_params = copy.deepcopy(params)
            local_net_state = copy.deepcopy(net_state)
            local_opt_state = optimizer_init(local_params)

            # 获取该客户端的校准先验
            local_priors = client_priors_list[client_id]

            indices = client_indices_list[client_id]
            num_samples = len(indices)
            num_batches = num_samples // BATCH_SIZE
            indices_jax = jnp.array(indices)

            rng, shuffle_key = jax.random.split(rng)
            shuffled_indices = jax.random.permutation(shuffle_key, indices_jax, independent=True)

            for b in range(num_batches):
                batch_idx = shuffled_indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
                batch_images = all_train_images[batch_idx]
                batch_labels = all_train_labels[batch_idx]

                rng, step_key = jax.random.split(rng)

                # 1. 生成校准后的对抗样本
                batch_images_adv = perturb_calfat(
                    local_params, local_net_state, net_forward_apply, step_key,
                    batch_images, batch_labels,
                    local_priors, args.cal_tau,
                    adv_eps_val, 2 * adv_eps_val / adv_iters, adv_iters,
                    attack_method=args.attack_method, is_training=True
                )

                # 2. 校准对抗训练 (Logit Adjusted Loss)
                loss, local_params, local_net_state, local_opt_state, acc = do_calfat_training_step(
                    local_params, local_net_state, net_forward_apply,
                    local_opt_state, optimizer_update, step_key,
                    batch_images_adv, batch_labels,
                    local_priors, args.cal_tau,
                    is_training=True
                )

            client_params_list.append(local_params)
            client_net_state_list.append(local_net_state)
            local_size_list.append(num_samples)

        # Aggregation
        params = fed_avg(client_params_list, local_size_list)
        net_state = fed_avg(client_net_state_list, local_size_list)

        # Evaluation
        should_eval_adv = (epoch % 5 == 0) or (epoch == args.epochs - 1)

        # Test 使用 Standard Test
        clean_acc, adv_acc = test(
            params, params, net_state, net_forward_apply, rng, test_loader,
            linear=False, make_adv_examples=should_eval_adv,
            attack_method=args.attack_method, adv_eps=args.eps, short=True
        )
        if not should_eval_adv: adv_acc = 0.0

        print(f"Ep {epoch} | Clean: {clean_acc:.2%} | Adv (CalFAT): {adv_acc:.2%}")

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, clean_acc, adv_acc])

        if args.constant_save and (epoch % 10 == 0):
            pickle.dump({'params': params, 'net_state': net_state},
                        open(f'./{args.save_path}/epoch_{epoch}.pkl', 'wb'))


if __name__ == '__main__':
    main()