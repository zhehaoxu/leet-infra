import torch


def _table(name, value, length):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dim() != 1:
        raise ValueError(f"{name} 必须是一维张量")
    if value.shape[0] != length:
        raise ValueError(f"{name} 的长度必须是 {length}，实际 {value.shape[0]}")
    v = value.to(torch.float64)
    if not torch.isfinite(v).all():
        raise ValueError(f"{name} 必须全为有限值")
    return v


def _used_positive(name, value, stop, table_name):
    """只校验会被读到的前 stop 个元素：末位是永远读不到的哨兵。"""
    window = value[:stop]
    if bool((window <= 0.0).any()):
        raise ValueError(f"{table_name} 在读得到区间内的每个元素都必须为正（前 {stop} 个）")
    if bool((window >= 1.0).any()):
        raise ValueError(f"{table_name} 在读得到区间内的每个元素都必须小于 1（前 {stop} 个）")


def _int(name, value, low=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python int")
    if value < low:
        raise ValueError(f"{name} 不合法（需 >= {low}）")
    return value


def _index_tensor(name, value, shape):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} 必须是 torch.Tensor")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise ValueError(f"{name} 必须是整数张量")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} 形状应为 {tuple(shape)}，实际 {tuple(value.shape)}")
    return value.to(torch.int64)


def _log_onehot(x_t, num_classes):
    onehot = torch.nn.functional.one_hot(x_t, num_classes).permute(0, 2, 1)
    return torch.log(onehot.to(torch.float64).clamp(min=1e-30))


def _known_class_rows(log_onehot_known, x_t, a, b, c, num_classes):
    """每个像素对应的那一行 q(x_t | x_{t-1 or x_0} = 已知类)，形状 (B, C-1, L)。"""
    rows = (log_onehot_known + a).logaddexp(b)
    masked = (x_t == num_classes - 1).unsqueeze(1).expand(-1, num_classes - 1, -1)
    return torch.where(masked, c, rows)


def solve(model_output, sample, timestep, at, ct, att, ctt, num_vec_classes, gumbel_uniform):
    """VQ-Diffusion 的掩码离散反扩散单步（diffusers `VQDiffusionScheduler`）。"""
    num_embed = _int("num_vec_classes", num_vec_classes, low=2)
    mask_class = num_embed - 1

    if not isinstance(model_output, torch.Tensor):
        raise ValueError("model_output 必须是 torch.Tensor")
    if model_output.dim() != 3:
        raise ValueError("model_output 必须是 (batch, num_vec_classes - 1, num_latent_pixels) 三维张量")
    if tuple(model_output.shape)[1] != mask_class:
        raise ValueError("model_output 的第 1 维必须等于 num_vec_classes - 1")
    if not torch.is_floating_point(model_output):
        raise ValueError("model_output 必须是浮点张量")
    if not torch.isfinite(model_output).all():
        raise ValueError("model_output 必须全为有限值")
    log_p_x_0 = model_output.to(torch.float64)

    batch, _, length = tuple(log_p_x_0.shape)
    x_t = _index_tensor("sample", sample, (batch, length))
    if bool((x_t < 0).any()) or bool((x_t > mask_class).any()):
        raise ValueError(f"sample 的取值必须落在 [0, {mask_class}] 内")

    if not isinstance(at, torch.Tensor) or not isinstance(ct, torch.Tensor):
        raise ValueError("at / ct 必须是 torch.Tensor")
    if at.dim() != 1 or ct.dim() != 1 or at.shape != ct.shape:
        raise ValueError("at 与 ct 必须是一维同长张量")
    num_train = int(at.shape[0])
    if num_train < 2:
        raise ValueError("at / ct 至少要有 2 个时间步")

    t = _int("timestep", timestep, low=0)
    if t >= num_train:
        raise ValueError(f"timestep 必须落在 [0, {num_train}) 内")

    a = _table("at", at, num_train)
    c = _table("ct", ct, num_train)
    acc = _table("att", att, num_train + 1)
    ccc = _table("ctt", ctt, num_train + 1)

    b = (1.0 - a - c) / mask_class
    bc = (1.0 - acc - ccc) / mask_class
    for name, v in (("at", a), ("ct", c)):
        if bool((v <= 0.0).any()) or bool((v >= 1.0).any()):
            raise ValueError(f"{name} 的每个元素都必须落在 (0, 1) 内")
    if bool((1.0 - a - c <= 0.0).any()):
        raise ValueError("at 与 ct 之和必须小于 1，单个转移矩阵要留出非零的 b = 1 - a - c")
    _used_positive("att", acc, num_train, "att")
    _used_positive("ctt", ccc, num_train, "ctt")
    if bool((bc[:num_train] <= 0.0).any()):
        raise ValueError("att 与 ctt 之和必须小于 1，累积转移矩阵要留出非零的 b = 1 - a - c")

    if not isinstance(gumbel_uniform, torch.Tensor):
        raise ValueError("gumbel_uniform 必须是 torch.Tensor")
    if not torch.is_floating_point(gumbel_uniform):
        raise ValueError("gumbel_uniform 必须是浮点张量")
    want = (batch, num_embed, length) if t > 0 else (batch, mask_class, length)
    if tuple(gumbel_uniform.shape) != want:
        raise ValueError(f"gumbel_uniform 形状应为 {want}（t=0 时比 t>0 少一个掩码类）")
    u = gumbel_uniform.to(torch.float64)
    if bool((u < 0.0).any()) or bool((u >= 1.0).any()):
        raise ValueError("gumbel_uniform 必须落在 [0, 1) 内")

    log_a, log_b, log_c = torch.log(a), torch.log(b), torch.log(c)
    log_acc, log_bc, log_ccc = torch.log(acc), torch.log(bc), torch.log(ccc)

    if t == 0:
        log_p_x_t_min_1 = log_p_x_0
    else:
        log_onehot = _log_onehot(x_t, num_embed)
        log_onehot_known = log_onehot[:, :-1, :]

        log_q_x_t_given_x_0 = _known_class_rows(
            log_onehot_known, x_t, log_acc[t], log_bc[t], log_ccc[t], num_embed
        )
        # 掩码行必须在丢弃它之前保存：它本身就是 q(x_t | x_{t-1} = mask)
        log_onehot_from_masked = log_onehot[:, -1, :].unsqueeze(1)
        log_q_t_given_x_t_min_1 = _known_class_rows(
            log_onehot_known, x_t, log_a[t], log_b[t], log_c[t], num_embed
        )

        q = log_p_x_0 - log_q_x_t_given_x_0
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp

        q = (q + log_acc[t - 1]).logaddexp(log_bc[t - 1])
        q = torch.cat((q, log_ccc[t - 1].expand(batch, 1, length)), dim=1)

        log_p_x_t_min_1 = q + torch.cat((log_q_t_given_x_t_min_1, log_onehot_from_masked), dim=1) + q_log_sum_exp

    gumbel_noise = -torch.log(-torch.log(u + 1e-30) + 1e-30)
    prev_sample = (log_p_x_t_min_1 + gumbel_noise).argmax(dim=1)

    return prev_sample, log_p_x_t_min_1
