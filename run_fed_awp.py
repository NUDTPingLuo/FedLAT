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

# 导入基础测试函数
from test_functions import perturb, test, loss_fn as standard_loss_fn
import data


# =============================================================================
# 1. AWP 核心逻辑: 权重扰动生成
# =============================================================================

def calc_awp_perturbation(params, net_state, net_fn, rng, adv_images, labels, gamma, is_training=True):
    """
    计算权重的对抗扰动 v.
    目标: max_v Loss(w + v, x_adv, y)
    约束: ||v|| <= gamma * ||w||
    """
    # 1. 计算关于权重的梯度
    # [修正] 添加 has_aux=True，因为 standard_loss_fn 返回 (loss, aux)
    # jax.grad 返回 (grads, aux)，我们只需要 grads
    grads, _ = jax.grad(standard_loss_fn, argnums=0, has_aux=True)(
        params, params, net_state, net_fn, rng, adv_images, labels,
        lin=False, is_training=is_training, centering=False
    )

    # 2. 计算扰动 v (根据 AWP 论文算法)
    # v = gamma * ||w|| * grad / ||grad||
    # 这里的 norm 是 layer-wise 的

    def get_layer_perturbation(g, w):
        # 避免除零
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g))) + 1e-12
        w_norm = jnp.sqrt(jnp.sum(jnp.square(w))) + 1e-12

        # AWP 更新公式: v = gamma * (w_norm / g_norm) * g
        v = gamma * (w_norm / g_norm) * g
        return v

    perturbation_v = jax.tree_util.tree_map(get_layer_perturbation, grads, params)

    return perturbation_v


def add_perturbation(params, perturbation_v):
    """ w_perturbed = w + v """
    return jax.tree_util.tree_map(lambda w, v: w + v, params, perturbation_v)


# =============================================================================
# 2. Training Step (包含 AWP 逻辑)
# =============================================================================

# static_argnums=(2, 4, 10) -> net_fn, optimizer_update, is_training
@functools.partial(jax.jit, static_argnums=(2, 4, 10))
def do_awp_training_step(params, net_state, net_fn, opt_state, optimizer_update, rng,
                         adv_images, labels, gamma, awp_warmup_done, is_training=True):
    """
    执行一步 Fed-AWP 训练
    """

    # --- AWP Step ---
    # 使用 lax.cond 处理条件逻辑

    def with_awp(_):
        # 1. 计算 v
        v = calc_awp_perturbation(params, net_state, net_fn, rng, adv_images, labels, gamma, is_training)
        # 2. w_proxy = w + v
        params_proxy = add_perturbation(params, v)
        return params_proxy

    def without_awp(_):
        return params  # 不做扰动

    # 动态选择是否使用 AWP
    params_proxy = lax.cond(awp_warmup_done, with_awp, without_awp, operand=None)

    # --- Gradient Step ---
    # 在 params_proxy (扰动后的权重) 上计算梯度
    [loss, lf_dict], grads = jax.value_and_grad(standard_loss_fn, has_aux=True)(
        params_proxy, params_proxy, net_state, net_fn, rng, adv_images, labels,
        lin=False, is_training=is_training, centering=False
    )

    net_state = lf_dict['net_state']
    acc = lf_dict['acc']

    # --- Update Original Weights ---
    # 关键点：用 params_proxy 算出的梯度，去更新原始的 params
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='model arch')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')

    # 攻击参数
    parser.add_argument('--attack_method', type=str, default='pgd', choices=['fgsm', 'pgd', 'cw'])
    parser.add_argument('--eps', type=float, default=8.00, help='Input perturbation budget (8/255)')

    # AWP 参数
    parser.add_argument('--awp_gamma', type=float, default=0.005, help='Weight perturbation size (gamma)')
    parser.add_argument('--awp_warmup', type=int, default=10, help='Epochs before enabling AWP')

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

    # [修正] 兼容性转置逻辑
    if train_images_np.ndim == 4 and train_images_np.shape[1] in [1, 3]:
        if train_images_np.shape[2] > train_images_np.shape[1]:
            print(f"Transposing data from {train_images_np.shape} to NHWC format.")
            train_images_np = np.transpose(train_images_np, (0, 2, 3, 1))

    all_train_images = jnp.array(train_images_np)
    all_train_labels = jnp.array(train_labels_np)

    # Non-IID Setup
    num_clients = 10
    alpha_dirichlet = 0.1
    distribute = 'IID' if args.is_iid else 'Non_IID'
    client_indices_list = get_client_indices(train_labels_np, num_clients, iid=args.is_iid, alpha=alpha_dirichlet)

    test_loader = data.get_loader(args.dataset, train=False, batch_size=100, shuffle=False)

    args.save_path = f'results_awp/{args.dataset}/{args.attack_method}_{distribute}_{args.save_path}'
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

    csv_file_path = os.path.join(args.save_path, f'result_awp.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'clean', 'adv_acc'])

    # =========================================================================
    # Fed-AWP Training Loop
    # =========================================================================
    print(f">>> Starting Fed-AWP Training (Gamma={args.awp_gamma}, Warmup={args.awp_warmup})")
    BATCH_SIZE = 128
    adv_eps_val = args.eps / 255.0
    adv_iters = 10

    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        awp_active = epoch >= args.awp_warmup
        if awp_active and epoch == args.awp_warmup:
            print(">>> AWP Activated! Starting Weight Perturbation...")

        client_params_list = []
        client_net_state_list = []
        local_size_list = []

        for client_id in range(num_clients):
            local_params = copy.deepcopy(params)
            local_net_state = copy.deepcopy(net_state)
            local_opt_state = optimizer_init(local_params)

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

                # 1. 生成输入对抗样本 (Inner Maximization on X)
                # 使用 is_training=True 确保 BN 状态正确
                batch_images_adv = perturb(
                    local_params, local_params, local_net_state, net_forward_apply, step_key,
                    batch_images, batch_labels,
                    adv_eps_val, 2 * adv_eps_val / adv_iters, adv_iters,
                    linear=False, centering=False, attack_method=args.attack_method,
                    is_training=True
                )

                # 2. 执行 AWP 训练步 (Maximization on W + Minimization on Loss)
                loss, local_params, local_net_state, local_opt_state, acc = do_awp_training_step(
                    local_params, local_net_state, net_forward_apply,
                    local_opt_state, optimizer_update, step_key,
                    batch_images_adv, batch_labels,
                    args.awp_gamma, awp_active,
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

        clean_acc, adv_acc = test(
            params, params, net_state, net_forward_apply, rng, test_loader,
            linear=False, make_adv_examples=should_eval_adv,
            attack_method=args.attack_method, adv_eps=args.eps, short=True
        )
        if not should_eval_adv: adv_acc = 0.0

        print(f"Ep {epoch} | Clean: {clean_acc:.2%} | Adv (AWP): {adv_acc:.2%}")

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, clean_acc, adv_acc])

        if args.constant_save and (epoch % 10 == 0):
            pickle.dump({'params': params, 'net_state': net_state},
                        open(f'./{args.save_path}/epoch_{epoch}.pkl', 'wb'))


if __name__ == '__main__':
    main()