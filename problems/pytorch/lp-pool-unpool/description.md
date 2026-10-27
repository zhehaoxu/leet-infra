# LpPool 与 maxpool 索引/unpool

## 题目描述

三种池化共用一套窗口几何（`floor((n + 2p − k)/s) + 1`），但各自埋着一个不同的坑：

- **LpPool**：输出是窗口内 \(|x|\) 的 \(p\) 范数 \(\big(\sum_i |x_i|^p\big)^{1/p}\)，不是均值、也不带符号。`p=1` 就是 \(|x|\) 求和（`avg_pool2d` 差一个 \(1/k^2\) 因子，原生 `F.lp_pool2d` 的 `p=1` 支路还是**带符号**求和），`p=∞` 才是 maxpool；题面把 \(p\) 限定在 \(p\ge 1\) 且有限，并对**每个元素取 \(|x|\)**——这是本仓库自定的确定性口径，判题会同时用带负输入探针它。
- **maxpool + indices**：除最大值外还要返回"最大值来自原图哪个位置"，索引是**未填充**原空间图上的行优先展平偏移（\(\text{row}\times W + \text{col}\)），并且并列最大时取行优先最前者。填充位补 \(-\infty\) 而不是 0，否则负数区域会被 0 顶掉。
- **maxunpool**：拿上面的 indices 把最大值写回源位置、其余位置为 0，构成 maxpool 的"稀疏逆"。它只在窗口不重叠时才有良定义的行为，所以题面要求**各维 stride ≥ kernel**，重叠配置直接抛错。

一个函数四种 mode，CPU、全程确定性、无随机数调用。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。`mode` 取 `"lppool2d" | "lppool1d" | "maxpool_idx" | "maxunpool"`，其他取值抛 ValueError；`kernel_size/stride/padding` 接受 int 或长度 1/2 的整数序列，`stride=None` 时逐维默认等于 kernel。
- **窗口几何（三维统一）**：输出边长 \(\;o = \lfloor (n + 2p_{\text{pad}} - k)/s \rfloor + 1\)，须 \(o\ge 1\)；`padding` 每边不得超过对应 kernel 维的一半（原生 `F.max_pool2d` 在此处抛 RuntimeError，本题统一为 ValueError）。
- **lppool**（1-D/2-D）：输出 \(= \big(\sum_{\text{窗口}} |x_i|^p\big)^{1/p}\)，零填充，`p` 须 \(\ge 1\) 且有限；输出与 `x` 同形状约定（1-D 时 \((N,C,ol)\)，2-D 时 \((N,C,oh,ow)\)）同 dtype。`lppool1d` 要求 `x` 恰 3 维且 kernel/stride/padding 逐维一致。
- **maxpool_idx**：返回 `(out, indices)` 二元组。`out` 语义同 `F.max_pool2d`；`indices` 为 int64、与 `out` 同形状，取值是最大值在**原图**（未填充）\(H\times W\) 上的行优先展平偏移。该 mode 完全忽略 `p`。
- **maxunpool**：输入是小图 \((N,C,h,w)\) 加 `indices`（形状与 `x` 一致、整数 dtype），输出 \((N,C,H,W)\)：每个窗口值写回 `indices` 指向的位置，其余为 0。`output_size=None` 时按 \(H=(h-1)s_h+k_h\) 逐维推断；显式给定时须逐维 \(\ge\) 小图尺寸。要求 `padding == 0` 且各维 `stride >= kernel`。
- **秩约束**：`lppool1d` 要 3-D，其余三种要 4-D，不符抛 ValueError。
- **错误约定**：本题所有非法入参（非法 mode、非法秩、`p` 非法、padding 超限、窗口放不下、indices 形状/dtype/越界、output_size 过小、重叠配置）**一律抛 ValueError**，与原生的 RuntimeError 口径不同，判题按 ValueError 断言。
- 返回新张量、不修改入参；`maxpool_idx` 的并列最大取行优先最前者。

## 示例

\(x\) 为 (1,1,3,3)：

\[x=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix},\qquad k=2,\ s=2,\ p=2\]

输出边长 \(o=\lfloor(3-2)/2\rfloor+1=1\)（第 3 行、第 3 列被 floor 丢弃，不参与任何窗口），唯一窗口是 \([1,2,4,5]\)：

- `lppool2d`：\(\sqrt{1^2+2^2+4^2+5^2}=\sqrt{46}\)，float32 实测 **6.7823300362**（数学值 \(6.7823299831\ldots\)，落在判题 \(10^{-6}\) 容差内）。
- `maxpool_idx`：`out = 5.0`；最大值 5 在原图坐标 \((1,1)\)，行优先展平偏移 \(1\times 3 + 1 = \) **4**，`indices = 4`。
- `maxunpool`（`output_size=(3,3)`、indices 取上一步结果）：\(3\times3\) 全 0，只有 \((1,1)\) 处为 5。

`maxunpool` 的这一步必须显式传 `output_size=(3,3)`：`output_size=None` 会按 \(H=(1-1)\times2+2=2\) 推出 \(2\times2\)，此时 indices 的 4 已越界并抛 ValueError——残边被 floor 丢弃时，反推公式给的是"最小可能输出"而不是原图尺寸。

## 约束

- 形状：`lppool2d`/`maxpool_idx` 的 `x` 为 \((N,C,H,W)\)、`lppool1d` 为 \((N,C,L)\)；`maxunpool` 的 `x` 是小图，`indices` 与小图同形状。
- dtype：`lppool*` 输出与 `x` 同 dtype（float32/float64）；`maxpool_idx` 的 `indices` 为 int64；`maxunpool` 输出与 `x` 同 dtype。
- 判题容差：float32 用 \(10^{-6}\)（`torch.allclose`，与原生 `F.lp_pool2d`/`F.max_pool2d` 对拍时实测最大偏差在 \(10^{-7}\) 量级），float64 用 \(10^{-12}\)；`indices` 用 `torch.equal` 精确比对。
- 无随机数；全零窗口（含零填充造成的窗口）输出必须是 0，而不是 NaN 或未定义值。
