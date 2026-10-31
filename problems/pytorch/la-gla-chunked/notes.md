# notes — la-gla-chunked（order 659）

## 回源

- 官方实现：`fla-org/flash-linear-attention`，`fla/ops/gla/naive.py::naive_recurrent_gla`（`h = h·exp(gk_i)[..., None] + k_i[...,None]·v_i[...,None,:]`、`o_i = (q_i·scale).sum(-2)`、`scale = K**-0.5`、`.float()` 化）。chunk 侧官方 `fla/ops/gla/chunk.py` 用 BT 块 + matmul 重排（含 GroupNorm/头拆分），本题只收**数学恒等式**：chunk 并行形式 ≡ 朴素递推，不收 kernel 分块细节。
- 论文：2312.06635（Gated Linear Attention Transformers with Hardware-Efficient Training）。
- 与 651 `la-gated-deltanet-recurrent` 划界：那是 delta rule（状态做减法更新 `h += k⊗(v − h·k)·β`，g 是门控衰减乘在 h 上），本题是纯 GLA（无 delta 减法项）且考的是 chunk 恒等式本身。

## 判题面

- solve 入参：q/k/v/g `[T,·]`（g `[T,K]`）、chunk_size、h0 `[K,V]` 可 None、scale 可 None（缺省 K^-0.5）。返回 o/final_state/chunk_states（每个 chunk 更新完的状态，账本）。
- 全程 float64（题面钉死）；oracle `test_cases._py` 独立纯 Python 朴素递推，容差 1e-8/1e-6。
- 锚点：① 单 chunk K=V=1 手算（`o1 = exp(S1−S0)·k0v0 + k1v1`）；② chunk=1 两段（段间项 `(q·exp(g1))·h1` 可观察）；③ h0 贡献 `q·scale·exp(g0)·h0`；④ 随机张量 T=7/chunk=3（含不均匀尾 chunk）与朴素递推对拍；⑤ chunk_size ∈ {1,2,4,6,9>T} 全部同解（恒等式与切法无关）；⑥ 缺省 scale 与 scale=1 的比值 = K^-0.5。
- 非法输入：q/k/g 形状不匹配、v 行数≠T、chunk_size ≤0/非整/bool、空序列、h0 形状、scale ≤0/inf/nan → ValueError。注意 `v[:, :1]` 是合法的 V=1 输入，**不**算非法。

## 实测踩坑（变异测试前的真 bug）

1. **torch 广播按尾维**：`torch.exp(gc[-1]) * h`（`[K] × [K,V]`）把衰减乘到了**列**（value 通道）上；K=V=2 时静默出错，靠 chunk=1 与朴素递推对拍暴露。修复：`torch.exp(gc[-1])[:, None] * h`（按 key 通道逐行）。该错误已做成变异 `row-decay-broadcast`。
2. **A 矩阵维度**：`g` 是 `[T,K]`，`exp(gc)[:, None, :]` 的三维修写容易把 `[l,K]` 又乘成 `[l,l,K]`；正确写法是两支先各自按 token 衰减（`qc ⊙ exp(gc)` 与 `kc ⊙ exp(−gc)`）再做 `[l,K]@[K,l]`。
3. 锚点张量必须 `dtype=torch.float64` 构造：默认 float32 会先把 `-ln2` 之类常量舍入，1e-12 级断言必炸。

## 变异记录（9 条全抓）

inter-no-decay（段间项丢 `exp(S)`）、intra-no-mask（段内泄未来）、intra-diag-drop（对角丢失）、state-no-end-decay（状态更新丢段末衰减）、state-k-no-decay（k 累积丢 `exp(S_end−S)`）、dec-end-from-zero（`exp(S_end−S)` 退化 `exp(−S)`，状态整体差 `exp(S_end)` 倍）、scale-drop（丢缺省 `K^-0.5`）、cumsum-suffix（前缀 cumsum 换后缀）、row-decay-broadcast（上述真 bug）。

## 红线

头拆分/GroupNorm/kernel 分块（BT 矩阵重排、inter/intra 融合）不出；batch/head 维不出（单序列单头口径）；与 `cumsum-scan`（前缀和）及 attention 系查重。
