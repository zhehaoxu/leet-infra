# 详解：梯度累积 + 全局范数裁剪 + 动量 SGD

## 出处与回源
- **梯度裁剪口径**取自 PyTorch 官方 `torch.nn.utils.clip_grad_norm_`。核实其源码 `clip_grads_with_norm_` / `get_total_norm`：把每个参数的 \(\|g_p\|_2\) 堆叠后取 `vector_norm(norm_type=2)`，即 \(\text{total}=\sqrt{\sum_p\|g_p\|_2^2}\)；裁剪系数 `clip_coef = max_norm / (total_norm + 1e-6)`，再 `clamp_(max=1.0)`，最后每个梯度 `mul_(clip_coef)`。本题**先对累积梯度求均值**再套用这一裁剪，与主流框架「累积后除以累积步数、再裁剪」一致。
- **梯度累积**是大 batch 训练的常规技巧（`accum_steps` 个微批次合成一次更新），非某篇论文专属。
- **动量 SGD**：Polyak（1964）动量法，PyTorch `SGD(momentum=...)` 采用 \(b\leftarrow\mu b+g,\ p\leftarrow p-lr\cdot b\)（缓冲区不含 \((1-\mu)\) 因子），本题沿用此约定。

## 核心流程（易错点全在顺序）
1. **累积 vs 更新的时机**：`step()` 每微批次加一次 \(a\)，但只有当 \(m\) 达到 `accum_steps` 才真正更新 \(p\)。中途返回时**绝不能**动 \(p\)、也不能清 \(a\)。
2. **先均值后裁剪**：全局范数用的是 \(\bar g=a/\text{accum}\) 的范数，不是 \(a\) 的范数。若忘了除以 `accum_steps`，裁剪系数会差 `accum_steps` 倍。
3. **全局（跨参数）范数**：裁剪系数 \(c\) 对**所有**参与参数共享一个标量（把各参数范数平方相加再开根号），不是逐参数各自裁剪。
4. **`+1e-6` 与 `clamp(max=1)`**：分母加 \(10^{-6}\) 防零除；\(c\) 上限 1 保证只在超阈时缩小、绝不放大。
5. **复位**：一次有效更新后清零 \(a\) 与 \(m\)；但动量缓冲 \(b\) **不清**，跨大批次持续累积。

## 对拍口径
判题用独立重算 `_oracle_clipaccum`（用 list 版、把裁剪系数写成 Python `float` 的 `min(...)` 而非张量 `clamp`，结构与参考解不同），在 `allclose(atol=1e-5, rtol=1e-5)` 下比对；实测逐元素**完全一致**（maxdiff=0）。五个检查：
- `check_example`：`accum=4`、`randn*2.0` 大梯度触发裁剪、12 微批次（3 次有效更新）后的具体参数值。
- `check_matches_oracle_clipped`：三种形状混合、`accum=3`、`momentum=0.9`、`randn*3.0`，与 `_oracle_clipaccum` 对拍。
- `check_partial_micro_no_update`：只喂 2 个微批次（`accum=3`）时参数完全不变，验证「中途返回不更新、不清累积器」。
- `check_none_grad_param_skipped`：全程无梯度的参数不被创建状态、值不变。
- `check_returns_optimizer_instance`：`solve` 返回 `torch.optim.Optimizer` 实例。

## 复杂度
每微批次 \(O(\text{numel})\) 累积；每个大批次额外 \(O(\text{numel})\) 求范数与更新。状态 \(a,b\)（两份，与动量 SGD 同级）。
