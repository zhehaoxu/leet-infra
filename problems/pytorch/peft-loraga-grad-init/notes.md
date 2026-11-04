# 详解：LoRA-GA 梯度近似的初始化

## 出处

- 论文：**LoRA-GA: Low-Rank Adaptation with Gradient Approximation**（arXiv:2407.05000，Wang, Yu, Li）。
- 官方实现：`Outsider565/LoRA-GA`（自定义 peft fork；核心在 `tuners/lora/layer.py` 的 update_layer 分支与 `utils/lora_ga_utils/`）。
- 关键差异（vs PiSSA/LoRA-DORA 一系）：LoRA-GA 初始化后把 \(W \leftarrow W-\eta\,BA\) **吸收进基座**，训练起点=预训练模型；PiSSA 则保留 \(W\) 不动、把主成分移进 adapter。加上第一步梯度对齐，等价于全微调的更新方向。

## 官方实现的逐行翻译

**grad_svd**（layer.py，逐行）：

```python
grad = self.kwargs["grad"].to(device).to(torch.float32)
U, S, V = torch.svd_lowrank(grad.float(), q=min(4 * lora_r, min(grad.shape)), niter=4)
V = V.T
```

```python
# 官方 layer 内两次 svd_lowrank 都没有 manual_seed（依赖全局 RNG 状态）；
# 判题域把 seed 显式化（与 808 LoRA-SB 的 get_svd_grad 不同——那里官方有 manual_seed(42)）。
```

**direction 四态**（layer.py，逐行）：

```python
if init_config.direction == "ArBr":
    B = U[:, 0: 2 * lora_r: 2]      # 偶数列
    A = V[1: 2 * lora_r: 2, :]      # 奇数行
elif init_config.direction == "A2rBr":
    B = U[:, :lora_r];  A = V[lora_r: 2 * lora_r, :]
elif init_config.direction == "ArB2r":
    B = U[:, lora_r: 2 * lora_r];  A = V[:lora_r, :]
elif init_config.direction == "random":
    random_list = random.sample(range(2 * lora_r), 2 * lora_r)   # Python random！
    indexes_A = random_list[0:lora_r]      # 先 A 行
    indexes_B = random_list[lora_r:2 * lora_r]
    B = U[:, indexes_B];  A = V[indexes_A, :]
```

**scale 四态**（layer.py，逐行）：

```python
scaling_factor = self.scaling["default"]
if init_config.scale == "gd":
    A = A / scaling_factor;  B = B / scaling_factor
elif init_config.scale == "unit":
    pass                                    # A,B 正交，不缩放
elif init_config.scale == "stable":
    m, n = grad.shape                        # m: feature_out
    gamma = init_config.stable_gamma
    B = B * m ** 0.25 / gamma ** 0.5
    A = A * m ** 0.25 / gamma ** 0.5
elif init_config.scale == "weightS":
    _, S, _ = torch.svd_lowrank(weight.data.float(), q=4 * lora_r, niter=4)
    S = S / self.scaling["default"]
    avg_s = torch.sqrt(S[:lora_r]).mean()
    B = B * avg_s;  A = A * avg_s
```

**offset 吸收**（layer.py，逐行）：

```python
offset = B @ A
...
offset *= scaling_factor
if hasattr(init_config, "norm_clip") and init_config.norm_clip:
    ratio = torch.max(torch.abs(weight.data)) / torch.max(torch.abs(offset))
    if ratio < 1:
        offset *= ratio
        A *= ratio ** 0.5
        B *= ratio ** 0.5
weight.data -= offset
self.lora_A[adapter_name].weight.data = A.contiguous()
self.lora_B[adapter_name].weight.data = B.contiguous()
```

```python
# norm_clip 不变量：(B*sqrt(ratio)) @ (A*sqrt(ratio)) * eta == offset * ratio
# —— A、B 各乘 sqrt(ratio) 后 BA 恰好放大 ratio 倍，与 offset 的缩放同步。
# ratio == 1 时两分支均乘 1（m-nc-lt 真等价剔除）。
```

## 判题域剔除与红线

- **剔除**：quantized 权重前向（get_float_weight/bitsandbytes）、bf16/fp32 dtype 分支、OffloadContext/Accelerator/distributed、weightS 的 weight svd_lowrank（改为显式入参）。
- **保留**：grad_svd 全流程、direction 四态（含 Python random 账）、scale 四态、offset 吸收与 norm_clip 不变量。
- **锚定**：S/U/Vt fixtures 逐位锚定；niter=0 vs 4 差异用例。

## 关键怪癖（回源逐行确认）

- **官方无 seed**：layer.py 的两次 svd_lowrank 都依赖全局 RNG；判题域显式化 seed（官方 svd_utils 层有 manual_seed(42) 的是 LoRA-SB）。
- **ArBr 是交错选取不是前后半**：`0:2r:2` 截断在 2r，不是 `0::2` 到末尾（沙箱验证过切片差异）。
- **random 用 Python random**（非 torch RNG），且 indexes_A 先抽——m-di-ab 交换顺序可抓。
- **stable 的 m 是 out_features**（B.size(0)，官方注释 m: feature_out）；不是 r 也不是 in_features。
- **weightS 除 η 发生在 sqrt 之前**：`S /= η` 再 `sqrt(S[:r]).mean()`。
- **gd 不是等变缩放**：A、B 同时 /=η 后 η·BA = UVᵀ/η——官方语义就是线性除，无需过度解释。
- **norm_clip 的 ratio==1 不可观测**：两分支都乘 1（真等价）。

## 变异测试（23/23，1 等价剔除）

- m-nc-lt（ratio<1 → ratio<=1）：ratio==1 时两分支均乘 1，真等价剔除。
- 23 个变异覆盖：q/niter/seed/V.T/float32/rank 校验、四方向切片/random 顺序/枚举、四 scale 因子/m 来源/S 除法/sqrt/枚举、offset 的 η/BA 形式/加减号/norm_clip 的 A 指数。

## 与已有题的划界

- **LoRA-SB（808）**：sign 梯度估计 + 非对称 U·s/Vᵀ 拆分 + r×r latent；LoRA-GA：原始梯度 SVD + **交错方向银行** + 四 scale + **offset 吸收进基座**（LoRA-SB 不动基座）。
- **TinyLoRA（795）/PVeRA（804）**：SVD 源与拆分方式完全不同（见各自笔记）。
- 题库无「offset 吸收/权重减法」账（grep 已查重）。
