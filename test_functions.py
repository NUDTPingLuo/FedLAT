import jax
import jax.numpy as jnp
import numpy as np
import functools
import optax
from jax import lax  # 引入 lax 用于高性能循环
from models import linear_forward


# =============================================================================
# 损失函数定义
# =============================================================================

@functools.partial(jax.jit, static_argnums=(3, 7, 8, 9))
def loss_fn(params, lin_params, state, net_fn, rng, images, labels, lin=False, is_training=True, centering=False):
    """
    标准交叉熵损失函数 (Cross Entropy Loss)
    用于常规训练、PGD/FGSM 以及 MIM 攻击 (最大化此损失)
    """
    if not lin:
        # Phase 1: 标准网络前向传播
        if centering:
            # (f_t - f_0)
            logits0, state = net_fn(lin_params, state, rng, images, is_training=is_training)
            logits1, state = net_fn(params, state, rng, images, is_training=is_training)
            logits = logits1 - logits0
        else:
            logits, state = net_fn(params, state, rng, images, is_training=is_training)
    else:
        # Phase 2: NTK 线性化前向传播
        logits, state = linear_forward(params, lin_params, state, net_fn, rng, images, is_training=is_training,
                                       centering=centering)

    labels_oh = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, labels_oh).mean()
    acc = jnp.mean(logits.argmax(1) == labels)
    return loss, {'net_state': state, 'acc': acc}


@functools.partial(jax.jit, static_argnums=(3, 7, 8, 9))
def cw_loss_fn(params, lin_params, state, net_fn, rng, images, labels, lin=False, is_training=False, centering=False):
    """
    C&W 攻击使用的 Margin Loss
    目标: max(max(logits_other) - logits_target, -kappa)
    """
    if not lin:
        if centering:
            logits0, state = net_fn(lin_params, state, rng, images, is_training=is_training)
            logits1, state = net_fn(params, state, rng, images, is_training=is_training)
            logits = logits1 - logits0
        else:
            logits, state = net_fn(params, state, rng, images, is_training=is_training)
    else:
        logits, state = linear_forward(params, lin_params, state, net_fn, rng, images, is_training=is_training,
                                       centering=centering)

    # 获取目标类别的 logit
    batch_size = logits.shape[0]
    target_logits = logits[jnp.arange(batch_size), labels]

    # 获取非目标类别的最大 logit
    # 技巧: 将目标位置设为负无穷，然后取 max
    logits_other = logits - jax.nn.one_hot(labels, logits.shape[-1]) * 1e9
    max_other_logits = jnp.max(logits_other, axis=1)

    # 最大化 (max_other - target)
    loss = jnp.mean(max_other_logits - target_logits)

    return loss, {'net_state': state}


# =============================================================================
# 辅助函数：范数投影
# =============================================================================

@functools.partial(jax.jit, static_argnums=(2))
def clamp_by_norm(x, r, norm='l_inf'):
    """将扰动限制在指定的范数球内"""
    if norm == 'l_2':
        norms = jnp.sqrt(jnp.sum(x ** 2, axis=[1, 2, 3], keepdims=True))
        # 避免除以 0
        factor = jnp.minimum(r / (norms + 1e-12), 1.0)
        return x * factor
    elif norm == 'l_inf':
        return jnp.clip(x, -r, r)


# =============================================================================
# 单步扰动函数 (用于 scan 内部调用)
# =============================================================================

def do_perturbation_step_l_inf(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha,
                               linear=False, centering=False, is_training=False):
    """L_inf 范数下的 PGD/FGSM 单步更新"""
    grads, _ = jax.grad(loss_fn, has_aux=True, argnums=5)(
        params, lin_params, net_state, net_fn, rng, images, labels,
        lin=linear, is_training=is_training, centering=centering
    )
    grads = jnp.sign(grads)
    images = images + alpha * grads
    images = jnp.clip(images, 0., 1.)

    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm='l_inf')
    images = images0 + d_images
    return images


def do_perturbation_step_cw_l2(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha,
                               linear=False, centering=False, is_training=False):
    """L_2 范数下的 C&W 单步更新"""
    # 使用 cw_loss_fn 计算梯度
    grads, _ = jax.grad(cw_loss_fn, has_aux=True, argnums=5)(
        params, lin_params, net_state, net_fn, rng, images, labels,
        lin=linear, is_training=is_training, centering=centering
    )
    # L2 归一化梯度
    grads_norm = jnp.sqrt(jnp.sum(grads ** 2, axis=[1, 2, 3], keepdims=True))
    grads = grads / (grads_norm + 1e-10)

    images = images + alpha * grads
    images = jnp.clip(images, 0., 1.)

    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm='l_2')
    images = images0 + d_images
    return images


def do_perturbation_step_mim(params, lin_params, net_state, net_fn, rng, images0, images, labels, eps, alpha,
                             momentum, decay_factor=1.0, linear=False, centering=False, is_training=False):
    """
    MIM (Momentum Iterative Method) 单步更新
    """
    grads, _ = jax.grad(loss_fn, has_aux=True, argnums=5)(
        params, lin_params, net_state, net_fn, rng, images, labels,
        lin=linear, is_training=is_training, centering=centering
    )

    # MIM 特性：对梯度进行 L1 归一化
    grad_norm = jnp.sum(jnp.abs(grads), axis=(1, 2, 3), keepdims=True)
    normalized_grads = grads / (grad_norm + 1e-12)

    # 更新动量
    new_momentum = decay_factor * momentum + normalized_grads

    # 使用动量方向更新图像
    images = images + alpha * jnp.sign(new_momentum)
    images = jnp.clip(images, 0., 1.)

    # 投影回 L_inf 球
    d_images = images - images0
    d_images = clamp_by_norm(d_images, eps, norm='l_inf')
    images = images0 + d_images

    return images, new_momentum


# =============================================================================
# 对抗样本生成主函数 (使用 lax.scan 提速)
# =============================================================================

# static_argnums: 3(net_fn), 9(iters), 10(linear), 11(centering), 12(attack_method), 13(is_training)
@functools.partial(jax.jit, static_argnums=(3, 9, 10, 11, 12, 13))
def perturb(params, lin_params, net_state, net_fn, rng, images, labels, eps, alpha, iters, linear=False,
            centering=False, attack_method='pgd', is_training=False):
    """
    生成对抗样本 (JIT + Scan 优化版)
    支持: 'fgsm', 'pgd', 'mim', 'cw'
    """
    images0 = images

    # --- 1. FGSM ---
    if attack_method == 'fgsm':
        # FGSM 不需要循环，直接运行一步
        return do_perturbation_step_l_inf(
            params, lin_params, net_state, net_fn, rng, images0, images, labels,
            eps=eps, alpha=eps, linear=linear, centering=centering, is_training=is_training
        )

    # --- 2. PGD (使用 lax.scan 循环) ---
    elif attack_method == 'pgd':
        # 随机初始化
        noise = jax.random.uniform(rng, images.shape, minval=-eps, maxval=eps)
        images = images + noise
        images = jnp.clip(images, 0., 1.)
        d_images = clamp_by_norm(images - images0, eps, norm='l_inf')
        images = images0 + d_images

        # 定义循环体函数：输入 (carry, x) -> 输出 (new_carry, y)
        def body_fun(carry, _):
            current_images = carry
            next_images = do_perturbation_step_l_inf(
                params, lin_params, net_state, net_fn, rng, images0, current_images, labels,
                eps, alpha, linear=linear, centering=centering, is_training=is_training
            )
            return next_images, None

        # 使用 lax.scan 执行循环
        final_images, _ = lax.scan(body_fun, images, None, length=iters)
        return final_images

    # --- 3. MIM (Momentum Iterative Method) ---
    elif attack_method == 'mim':
        # 随机初始化
        noise = jax.random.uniform(rng, images.shape, minval=-eps, maxval=eps)
        images = images + noise
        images = jnp.clip(images, 0., 1.)
        d_images = clamp_by_norm(images - images0, eps, norm='l_inf')
        images = images0 + d_images

        # 初始化动量
        momentum = jnp.zeros_like(images)
        decay_factor = 1.0

        def body_fun_mim(carry, _):
            current_images, current_momentum = carry
            next_images, next_momentum = do_perturbation_step_mim(
                params, lin_params, net_state, net_fn, rng, images0, current_images, labels,
                eps, alpha, current_momentum, decay_factor,
                linear=linear, centering=centering, is_training=is_training
            )
            return (next_images, next_momentum), None

        (final_images, final_momentum), _ = lax.scan(body_fun_mim, (images, momentum), None, length=iters)
        return final_images

    # --- 4. C&W (Carlini & Wagner L2 PGD) ---
    elif attack_method == 'cw':
        # 随机初始化 (L2)
        noise = jax.random.normal(rng, images.shape) * 0.1
        images = images + noise
        images = jnp.clip(images, 0., 1.)

        # 这里你可以选择固定 eps_l2 或者使用传入的 eps（如果调用时做好了转换）
        eps_l2 = 1.0

        def body_fun_cw(carry, _):
            current_images = carry
            next_images = do_perturbation_step_cw_l2(
                params, lin_params, net_state, net_fn, rng, images0, current_images, labels,
                eps_l2, alpha, linear=linear, centering=centering, is_training=is_training
            )
            return next_images, None

        final_images, _ = lax.scan(body_fun_cw, images, None, length=iters)
        return final_images

    else:
        raise ValueError(f"Unknown attack method: {attack_method}")


# =============================================================================
# 测试函数
# =============================================================================

def test(params, lin_params, state, net_fn, rng, test_loader, linear=False, make_adv_examples=False, centering=False,
         attack_method='pgd', return_examples=False, short=False, return_components=False, adv_eps=4):
    """
    测试模型准确率 (Clean & Adversarial)
    """
    adv_eps_val = adv_eps / 255.0

    n_correct = 0
    n_total = 0
    n_correct_adv = 0

    print(f"Testing... Mode: {'Linear' if linear else 'Standard'}, Attack: {attack_method}")

    # 用于收集返回数据
    adv_examples = []
    predictions = []
    adv_predictions = []
    components = []
    linear_components = []
    adv_components = []
    adv_linear_components = []

    for i, (images, labels) in enumerate(test_loader):
        # 转换数据格式: PyTorch (NCHW) -> JAX (NHWC)
        images = np.array(np.transpose(images.cpu().numpy(), [0, 2, 3, 1]))
        labels = labels.cpu().numpy()

        # --- 1. Clean Evaluation ---
        if linear:
            logits, return_dict = linear_forward(params, lin_params, state, net_fn, rng, images, is_training=False,
                                                 centering=centering, return_components=True)
            if return_components:
                components.append(return_dict['f'])
                linear_components.append(return_dict['df'])
        else:
            if centering:
                logits0, _ = net_fn(lin_params, state, rng, images, is_training=False)
                logits1, _ = net_fn(params, state, rng, images, is_training=False)
                logits = logits1 - logits0
            else:
                logits, _ = net_fn(params, state, rng, images, is_training=False)

        n_correct += np.sum(logits.argmax(1) == labels)
        if return_examples:
            predictions.append(logits.argmax(1))

        # --- 2. Adversarial Evaluation ---
        if make_adv_examples:
            # 设定攻击参数
            iters = 20  # 默认 20 步
            alpha = 2 * adv_eps_val / iters

            if attack_method == 'fgsm':
                pass
            elif attack_method == 'mim':
                pass
            elif attack_method == 'cw':
                iters = 50
                alpha = 0.01

            # 生成对抗样本
            # 注意：测试阶段 is_training=False
            adv_images = perturb(
                params, lin_params, state, net_fn, rng, images, labels,
                eps=adv_eps_val, alpha=alpha, iters=iters,
                linear=linear, centering=centering, attack_method=attack_method,
                is_training=False
            )
        else:
            adv_images = images

        if return_examples:
            adv_examples.append(adv_images)

        # 对抗样本前向传播
        if linear:
            logits_adv, return_dict = linear_forward(params, lin_params, state, net_fn, rng, adv_images,
                                                     is_training=False, centering=centering, return_components=True)
            if return_components:
                adv_components.append(return_dict['f'])
                adv_linear_components.append(return_dict['df'])
        else:
            if centering:
                logits0, _ = net_fn(lin_params, state, rng, adv_images, is_training=False)
                logits1, _ = net_fn(params, state, rng, adv_images, is_training=False)
                logits_adv = logits1 - logits0
            else:
                logits_adv, _ = net_fn(params, state, rng, adv_images, is_training=False)

        n_correct_adv += np.sum(logits_adv.argmax(1) == labels)
        if return_examples:
            adv_predictions.append(logits_adv.argmax(1))

        n_total += len(labels)

        # 简短测试 (只测一部分)
        if short and i >= 9:  # 测试 10 个 batch
            break

    print(f"Clean Acc: {n_correct / n_total:.4f}")
    print(f"Dirty Acc ({attack_method}): {n_correct_adv / n_total:.4f}")

    # 结果打包返回
    if return_examples:
        adv_examples = np.concatenate(adv_examples, 0)
        predictions = np.concatenate(predictions)
        adv_predictions = np.concatenate(adv_predictions)

        if return_components:
            components_clean = {'f': np.concatenate(components), 'df': np.concatenate(linear_components)}
            components_dirty = {'f': np.concatenate(adv_components), 'df': np.concatenate(adv_linear_components)}
            return n_correct / n_total, n_correct_adv / n_total, adv_examples, predictions, adv_predictions, components_clean, components_dirty

        return n_correct / n_total, n_correct_adv / n_total, adv_examples, predictions, adv_predictions

    return n_correct / n_total, n_correct_adv / n_total