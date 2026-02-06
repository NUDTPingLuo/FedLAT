import jax
import haiku as hk
import jax.numpy as jnp
from jax.example_libraries import optimizers
import torch
import torchvision
import torchvision.transforms as transforms
# Removed unused Dataset imports since we use raw arrays
import numpy as np
import neural_tangents as nt
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

# 导入优化后的测试和扰动函数
from test_functions import perturb, test, loss_fn
import data


# =============================================================================
# JAX 训练 Step (JIT 编译)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(3, 5, 9, 10))
def do_training_step(params, lin_params, net_state, net_fn, opt_state, optimizer_update, rng, images, labels,
                     is_training=True, centering=False):
    """Phase 1: 标准 SGD 训练步骤"""
    [loss, lf_dict], grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, lin_params, net_state, net_fn, rng, images, labels, lin=False, is_training=is_training,
        centering=centering
    )
    net_state = lf_dict['net_state']
    acc = lf_dict['acc']
    updates, opt_state = optimizer_update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, net_state, opt_state, acc


@functools.partial(jax.jit, static_argnums=(3, 5, 9, 10))
def do_training_step_linear(params, lin_params, net_state, net_fn, opt_state_lin, optimizer_lin_update, rng, images,
                            labels, centering=False, is_training=False):
    """Phase 2: 线性化 (Linearized) 训练步骤"""
    [loss, lf_dict], grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=1)(
        params, lin_params, net_state, net_fn, rng, images, labels, lin=True, centering=centering,
        is_training=is_training
    )
    net_state = lf_dict['net_state']
    acc = lf_dict['acc']
    updates, opt_state_lin = optimizer_lin_update(grads, opt_state_lin, lin_params)
    lin_params = optax.apply_updates(lin_params, updates)
    return loss, params, lin_params, net_state, opt_state_lin, acc


# =============================================================================
# 数据处理工具 (GPU 优化版)
# =============================================================================

def get_client_indices(labels_np, num_clients, iid=True, alpha=0.1):
    """
    生成每个客户端的数据索引列表。
    仅在 CPU 上处理索引，不移动实际数据。
    """
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
            # Dirichlet 分割
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            # 计算分割点
            split_points = (np.cumsum(proportions) * len(label_indices[c])).astype(int)[:-1]
            split = np.split(label_indices[c], split_points)
            for i, part in enumerate(split):
                client_indices[i].extend(part)

        for i in range(num_clients):
            np.random.shuffle(client_indices[i])  # 打乱内部顺序
            indices_list.append(np.array(client_indices[i]))

    return indices_list


def fed_avg(param_list, local_size_list):
    """FedAvg 聚合算法"""
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
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: cifar10/cifar100/tinyimagenet')
    parser.add_argument('--model', type=str, default='resnet18', help='model arch')
    parser.add_argument('--standard_epochs', type=int, default=100, help='Phase 1 epochs')
    parser.add_argument('--linear_epochs', type=int, default=100, help='Phase 2 epochs')
    parser.add_argument('--loaders', type=str, default='CC',
                        help='Training mode. e.g., "AC" means Phase1 Adv, Phase2 Clean')
    parser.add_argument('--attack_method', type=str, default='pgd', choices=['fgsm', 'pgd', 'mim', 'cw'],
                        help='Attack method for Adv Training and Testing')
    parser.add_argument('--eps', type=float, default=4.00, help='Perturbation budget (scaled by 1/255 inside)')
    parser.add_argument('--second_lr', type=float, default=0.01, help='Phase 2 Learning Rate')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='Result save path')
    parser.add_argument('--constant_save', action='store_true', help='Save freq in Phase 1')
    parser.add_argument('--constant_save_linear', action='store_true', help='Save freq in Phase 2')
    parser.add_argument('--centering', action='store_true', help='Use centered dynamics in Phase 2')
    parser.add_argument('--loose_bn_second', action='store_true', help='Unfreeze BN in Phase 2')
    parser.add_argument('--do_standard_second', action='store_true', help='Use standard dynamics in Phase 2 (Ablation)')
    parser.add_argument('--skip_first_test', action='store_true')
    parser.add_argument('--skip_second_test', action='store_true')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--base_model_path', type=str, default='')
    parser.add_argument('--is_iid', type=bool, default=True)

    args = parser.parse_args()

    # --- GPU 数据加载 (提速核心) ---
    print(">>> Loading Data to GPU...")
    # 1. 获取原始 Numpy 数据 (通常是 NCHW)
    train_images_np, train_labels_np = data.get_data_and_labels(args.dataset)
    # 2. 转换为 NHWC (JAX 首选格式)
    # [修正]：检测 NCHW 格式并转置，无论它是 1通道(FMNIST/MNIST) 还是 3通道(CIFAR)
    if train_images_np.ndim == 4 and train_images_np.shape[1] in [1, 3]:
        # 只有当第2维是1或3，且第3、4维比第2维大得多时，才认为是 NCHW
        if train_images_np.shape[2] > train_images_np.shape[1]:
            print(f"Transposing data from {train_images_np.shape} to NHWC format.")
            train_images_np = np.transpose(train_images_np, (0, 2, 3, 1))

    # 3. 转换为 JAX Array (直接存入 GPU 显存)
    all_train_images = jnp.array(train_images_np)
    all_train_labels = jnp.array(train_labels_np)
    print(f"Data Loaded: {all_train_images.shape}")

    # 4. 获取客户端索引划分
    num_clients = 10
    alpha_dirichlet = 0.1
    distribute = 'IID' if args.is_iid else 'Non_IID'

    # 使用 Numpy 版本的数据进行索引划分 (CPU 计算)
    client_indices_list = get_client_indices(train_labels_np, num_clients, iid=args.is_iid, alpha=alpha_dirichlet)

    # 5. 准备测试集
    # 这里我们保留 DataLoader，因为测试频率低，瓶颈不在这里
    test_loader = data.get_loader(args.dataset, train=False, batch_size=100, shuffle=False)

    args.save_path = 'results/' + args.dataset + '/' + args.attack_method + '_' + distribute + '_' + args.save_path
    os.makedirs(args.save_path, exist_ok=True)

    rng = jax.random.PRNGKey(args.random_seed)
    print(f"Random Seed: {args.random_seed}, Attack Method: {args.attack_method}")

    # --- 模型初始化 ---
    net_forward_init, net_forward_apply = models.get_model(args.model, data.get_n_classes(args.dataset))

    # 使用 GPU 上的第一张图做初始化 dummy
    dummy_input = all_train_images[0:1]
    params, net_state = net_forward_init(rng, dummy_input, is_training=True)
    lin_params = copy.deepcopy(params)

    optimizer_init, optimizer_update = optax.chain(optax.sgd(0.01, momentum=0.9))
    opt_state = optimizer_init(params)

    # 加载模型逻辑
    if len(args.base_model_path) > 0:
        print('Loading from saved model...')
        args.base_model_path = 'results/' + args.dataset + '/' + args.attack_method + '_' + distribute + '_' + args.base_model_path
        with open('./{}'.format(args.base_model_path), 'rb') as f:
            checkpoint = pickle.load(f)
        params = checkpoint['params']
        lin_params = checkpoint['lin_params']
        net_state = checkpoint['net_state']

    csv_file_path = os.path.join(args.save_path, f'result_{args.attack_method}.csv')
    with open(csv_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'clean', 'adv_acc'])

    # =========================================================================
    # Phase 1: Standard Training Loop
    # =========================================================================
    print(">>> Starting Phase 1: Standard Dynamics")

    # 批大小
    BATCH_SIZE = 128

    for epoch in range(args.standard_epochs):
        print(f'Federated Standard Epoch {epoch}')
        client_params_list = []
        client_net_state_list = []
        local_size_list = []

        # 遍历客户端
        for client_id in range(num_clients):
            local_params = copy.deepcopy(params)
            local_net_state = copy.deepcopy(net_state)
            local_opt_state = optimizer_init(local_params)

            # 获取该客户端的数据索引 (CPU)
            indices = client_indices_list[client_id]
            num_samples = len(indices)
            num_batches = num_samples // BATCH_SIZE

            # 转为 JAX array 以便在 GPU 上索引 (Zero-Copy 关键)
            indices_jax = jnp.array(indices)

            # 打乱数据
            rng, shuffle_key = jax.random.split(rng)
            shuffled_indices = jax.random.permutation(shuffle_key, indices_jax, independent=True)

            for b in range(num_batches):
                # 1. 直接在 GPU 上切片获取 Batch (极速)
                batch_idx = shuffled_indices[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
                batch_images = all_train_images[batch_idx]
                batch_labels = all_train_labels[batch_idx]

                rng, step_key = jax.random.split(rng)

                # 2. 对抗训练逻辑
                if args.loaders[0] == 'A' or (args.loaders[0] == 'F' and epoch >= 50):
                    adv_eps_val = args.eps / 255
                    iters = 20
                    batch_images = perturb(
                        params, lin_params, net_state, net_forward_apply, step_key,
                        batch_images, batch_labels, adv_eps_val, 2 * adv_eps_val / iters, iters,
                        linear=False, centering=False, attack_method=args.attack_method,
                        is_training=True  # <--- 修改点 1：Phase 1 训练时开启 Training 模式
                    )

                # 3. 训练步
                loss, local_params, local_net_state, local_opt_state, acc = do_training_step(
                    local_params, lin_params, local_net_state, net_forward_apply,
                    local_opt_state, optimizer_update, step_key, batch_images, batch_labels
                )

            client_params_list.append(local_params)
            client_net_state_list.append(local_net_state)
            local_size_list.append(num_samples)

        # 聚合
        params = fed_avg(client_params_list, local_size_list)
        net_state = fed_avg(client_net_state_list, local_size_list)

        # 保存
        if args.constant_save:
            if epoch == 50 or epoch == 100:
                pickle.dump({'params': params, 'lin_params': lin_params, 'net_state': net_state},
                            open(f'./{args.save_path}/phase1_epoch_{epoch}.pkl', 'wb'))

        # 测试策略
        if args.skip_first_test:
            clean_acc, adv_acc = 0, 0
        else:
            # 只有在需要时才开启 make_adv_examples=True (慢)
            # 测试时 is_training 默认为 False，符合评估标准
            clean_acc, adv_acc = test(
                params, lin_params, net_state, net_forward_apply, rng, test_loader,
                linear=False, make_adv_examples=True,
                attack_method=args.attack_method, adv_eps=args.eps, short=True
            )

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, clean_acc, adv_acc])

        print(f"Ep {epoch} | Clean: {clean_acc:.2%} | Adv ({args.attack_method}): {adv_acc:.2%}")

    # =========================================================================
    # Phase 2: Linearized Training Loop
    # =========================================================================
    print(">>> Starting Phase 2: Linearized Dynamics")

    lin_params = copy.deepcopy(params)
    optimizer_lin_init, optimizer_lin_update = optax.chain(optax.sgd(args.second_lr, momentum=0.9))

    for epoch in range(args.linear_epochs):
        print(f'Federated Linear Epoch {epoch}')
        client_lin_params_list = []
        local_size_list = []

        for client_id in range(num_clients):
            local_params = copy.deepcopy(params)  # Fixed w_0
            local_lin_params = copy.deepcopy(lin_params)  # Trainable
            local_net_state = copy.deepcopy(net_state)
            local_opt_state_lin = optimizer_lin_init(local_lin_params)

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

                if args.loaders[1] == 'A' or (args.loaders[1] == 'F' and epoch >= 50):
                    adv_eps_val = args.eps / 255
                    iters = 20
                    # Linear 模式下的 Perturb
                    batch_images = perturb(
                        local_params, local_lin_params, local_net_state, net_forward_apply, step_key,
                        batch_images, batch_labels, adv_eps_val, 2 * adv_eps_val / iters, iters,
                        linear=True, centering=args.centering, attack_method=args.attack_method,
                        is_training=True  # <--- 修改点 2：Phase 2 训练时开启 Training 模式
                    )

                    if args.do_standard_second:
                        loss, local_params, local_net_state, _, acc = do_training_step(
                            local_params, local_lin_params, local_net_state, net_forward_apply,
                            None, None, step_key, batch_images, batch_labels,
                            is_training=args.loose_bn_second, centering=args.centering
                        )
                    else:
                        loss, local_params, local_lin_params, local_net_state, local_opt_state_lin, acc = do_training_step_linear(
                            local_params, local_lin_params, local_net_state, net_forward_apply,
                            local_opt_state_lin, optimizer_lin_update, step_key, batch_images, batch_labels,
                            centering=args.centering, is_training=args.loose_bn_second
                        )
                else:
                    loss, local_params, local_lin_params, local_net_state, local_opt_state_lin, acc = do_training_step_linear(
                        local_params, local_lin_params, local_net_state, net_forward_apply,
                        local_opt_state_lin, optimizer_lin_update, step_key, batch_images, batch_labels,
                        centering=args.centering, is_training=args.loose_bn_second
                    )

            client_lin_params_list.append(local_lin_params)
            local_size_list.append(num_samples)

        # 聚合
        lin_params = fed_avg(client_lin_params_list, local_size_list)

        if args.constant_save_linear:
            if epoch == 50 or epoch == 100:
                pickle.dump({'params': params, 'lin_params': lin_params, 'net_state': net_state},
                            open(f'./{args.save_path}/phase2_epoch_{epoch}.pkl', 'wb'))

        # Evaluation
        if args.skip_second_test:
            clean_acc, adv_acc = 0, 0
        else:
            is_linear_test = not args.do_standard_second
            clean_acc, adv_acc = test(
                params, lin_params, net_state, net_forward_apply, rng, test_loader,
                linear=is_linear_test, make_adv_examples=True,
                centering=args.centering,
                attack_method=args.attack_method, adv_eps=args.eps, short=True
            )

        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.standard_epochs + epoch, clean_acc, adv_acc])

        print(f"Ep {epoch} | Clean: {clean_acc:.2%} | Adv ({args.attack_method}): {adv_acc:.2%}")


if __name__ == '__main__':
    main()