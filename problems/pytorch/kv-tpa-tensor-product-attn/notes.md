# 解题说明 · TPA 张量积注意力的因子收缩与掩码

## 一句话

TPA 把每个 token 的 K/V 拆成「逐头系数 × 跨头共享基」两个张量积因子，decode 时先算 `bq·bk` 的小块再收缩，永远不物化每头的稠密 K——本题复刻它的 CPU 语义（sigmoid 门 + 四个 scale + 三种掩码）。

## 算法步骤

给定 `aq (B,N,H,R)`、`ak (B,M,H,S)`、`av (B,M,H,S_V)`、`bq (B,N,R,D)`、`bk (B,M,D,S)`、`bv (B,M,E,S_V)`：

1. 门控：\(\tilde a=\sigma(a)\)，逐元素 sigmoid；
2. 收缩出 q/k/v：

\[
q=\tilde a^q\cdot b^q\cdot s_q\ (B,N,H,D),\quad
k=\tilde a^k\cdot b^k\cdot s_k\ (B,M,H,D),\quad
v=\tilde a^v\cdot b^v\cdot s_v\ (B,M,H,E)
\]

3. 打分 \(S=q\cdot k\cdot s+\text{bias}\)，形状 `(B,H,N,M)`；
4. `prob = softmax(S, dim=M)`，`out = prob · v`，形状 `(B,N,H,E)`。

默认 \(s_q=1/R\)、\(s_k=1/S\)、\(s_v=1/S_V\)、\(s=D^{-1/2}\)。`is_causal` 与 `attn_mask` 互斥；causal 是 `tril(N,M)`（左上对齐），bool mask 里 `True=保留`。

## 源码级细节

上游 `decode/tpa_flash_kernel_prefill_mask.py`：

- `tpa_decode_naive_torch`（L970-1037）先物化 q/k/v：`torch.einsum("b n h r, b n r d -> b n h d", torch.sigmoid(aq), bq) * scale_q`，key 用 `"b m h s, b m d s-> b m h d"`，value 用 `"b m h s, b m e s-> b m h e"`；再 `"b n h d, b m h d -> b h n m"` 打分。
- `tpa_decode_torch`（L1039-1106）是不物化版本：先 `score1 = einsum("b n r d, b m d s -> b n m r s", bq, bk)`，再 `score2 = einsum("b n h r, b n m r s -> b n m h s", sigmoid(aq), score1)`，最后 `score3 = einsum("b n m h s, b m h s -> b h n m", score2, sigmoid(ak))`，softmax 时统一乘 `scale_q * scale_k * scale`；输出侧 `prob` 先乘 `sigmoid(av)` 再与 `bv` 收缩，最后乘 `scale_v`。
- 两条路径在数学上等价（sigmoid 因子可以结合进 q/k/v），本题只按等价语义判分；`score1` 的中间形状 `(B,N,M,R,S)` 才是「不物化 K」的代价来源。
- 掩码：`attn_bias = torch.zeros(n, m, dtype=aq.dtype)`，只有 `(N,M)` 两个轴，靠广播覆盖 batch/head；`is_causal` 时上游用 `assert attn_mask is None`，本题把它改成 `ValueError`。
- bool mask 的语义是 `masked_fill_(attn_mask.logical_not(), -inf)`，即 `True` 保留；浮点 mask 直接 `attn_mask + attn_bias`。
- 本文件的默认值比无掩码版多两个：`scale_k=1/S`、`scale_v=1/S_V`（`decode/flashtpa_decode_torch.py` 里它们固定为 1，且没有 sigmoid、`ak` 是 `(B,M,H)` 单秩）。两套 API 不能混用，题面已钉死带掩码的这一套。

## 确定性边界

- 全程 float64：sigmoid、四个 scale、softmax 与两次输出收缩都在 float64 下完成，最后才转回输入 dtype；float32 输入按 `3e-5` 容差比对。
- `torch.einsum` / numpy 的收缩顺序可能不同，末位会有 1e-16 级差异；测试用独立 numpy 实现按元素对拍，不要求逐位相等。
- 被 causal 或 mask 遮掉的位置，`exp(-inf)=0`，所以 `prob` 里恰为 0，测试直接断言这一点。
- 全 `-inf` 的行（query 一个 key 都看不到）会让 softmax 产生 NaN；题面把这种输入排除在测试之外，实现不必额外兜底。
- 无随机数、无校准、无可训练参数；同一输入永远同一输出。

## 与其他题的区别

- `flash-attention-tile`：稠密 Q/K/V 的分块在线 softmax，重点是分块与 rescale；本题没有分块循环，重点是**因子分解后的收缩顺序与 scale 落点**。
- `grouped-query-attention`：多个 query head 共享同一份 K/V head，是 head 轴的复用；TPA 是每 token 的 K/V 本身被拆成两个因子，且基 `b` 在 **head 轴**上共享。
- `kv-xkv-layer-svd-share`：用 SVD 在层间共享奇异向量，是离线的低秩压缩；TPA 是模型结构自带的双因子参数化，不跑 SVD。
- `causal-mask`：只造掩码；本题把掩码、门控、scale 与两次收缩一起考。

## 验证清单

- `check_example`：题面 2×3 单头例子，核对 `prob` 两行与 `out` 的数值，并确认 causal 外概率恰为 0。
- `check_matches_oracle`：5 组不同形状（含 `N != M`、float32/float64、有无 causal）对拍独立 numpy 实现。
- `check_sigmoid_gate_is_required`：把 `aq/ak/av` 平移后，跳过 sigmoid 的结果必须对不上，防止只做线性收缩。
- `check_scales_are_required`：默认 `1/R`、`1/S`、`1/S_V` 各自生效（替换成 1 会改变输出），并验证四个显式 scale 覆盖。
- `check_causal_rectangular`：`N=3, M=5` 时第 `i` 行只看 `j<=i`，行和为 1，第 0 行概率为 one-hot。
- `check_bool_and_float_masks_agree`：bool `True=保留` 与等价 `0/-inf` 加法掩码逐位一致，被遮位置概率为 0。
- `check_shapes_and_no_mutation`：输出形状/dtype 正确、概率行和为 1、六个输入都没被就地修改。
- `check_is_causal_excludes_mask`：`is_causal=True` 同时给 `attn_mask` 必须抛 `ValueError`。
- `check_validation`：R/M/E 不一致、dtype 混杂、非四维、mask 形状/类型错误、非法 scale、非 bool `is_causal` 均抛 `ValueError`。
