# 详解：Cautious 符号一致性掩码（Cautious Optimizer）

## 出处
Liang, Zhang, Kwon, et al., *Cautious Optimizers: Improving Training with One Line of Code*（arXiv:2411.16085）。掩码算子口径与 kozistr/pytorch-optimizer 基类里的 `apply_cautious(update, grad)` 一致：

```
mask = (update * grad > 0).to(grad.dtype)
mask.mul_(mask.numel() / (mask.sum() + 1))
update.mul_(mask)
```

## 直觉
动量型更新 \(u_t\) 是历史梯度的 EMA，个别坐标上它可能和最新梯度 \(g_t\) **符号相反**——等于“顶着当前梯度往回走”。作者统计发现这类冲突坐标占比可观，抑制它们能稳定并加速训练。做法是：

1. 只保留 \(u_t\odot g_t>0\)（与当前梯度同号）的坐标，其余置 0；
2. 由于清零会削弱整体幅度，用 \(\dfrac{N}{\sum M+1}\) 重新缩放（\(N\) 为元素数、\(\sum M\) 为存活坐标数；\(+1\) 防全部冲突时除零），使被保留的方向“补回”被删掉的量级。

对 AdamW 而言，\(u\) 取的是**尚未乘 `lr` 的更新方向** \(\hat m/\text{denom}\)，掩码作用在它上面，最后再 \(p\leftarrow p-lr\cdot(u\odot M)\)。本题基座用 AdamW（解耦权重衰减、带偏差修正），与 arXiv:1711.05101 对齐。

## 与参考解对齐的易错点
1. **`+1` 在分母**：缩放因子是 \(N/(\sum M+1)\)，不是 \(N/\sum M\)。
2. **掩码用的是原始梯度 \(g\)**（非 \(m\)）：判据 \(u\odot g>0\)。解耦权重衰减只作用在 \(p\) 上、不改 \(g\)，故 \(g\) 即 `p.grad`。
3. **\(N\)** 是当前参数张量的元素个数（`mask.numel()`），逐参数计算。

## 对拍口径
判题用独立向量化重算（`_oracle_cautious`）实现同样的 AdamW 方向 + 掩码；另设 `check_cautious_mask_changes_result`：与“不加掩码的 AdamW”比较，断言掩码确实改变了结果（即确有冲突坐标被抑制）。`grad is None` 参数连权重衰减也跳过。

## 复杂度
与 AdamW 相同（\(O(\text{numel})\)、两份状态），额外一次逐元素比较与求和。
