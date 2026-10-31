# 解题说明 · CapKV 的 leverage score

## 一句话

两段式打分：先用**平均未来 RoPE** 把 query 变成一个 anchor，得到每个 token 的 query 相关性权重；再在 value 空间的**容量矩阵**上算 leverage score，两者相乘即为 token 分数。

## leverage score 在测什么

`A = I + Σ (w_j + eps) v_j v_j^T` 把所有 value 的加权外积累加成一个"容量"矩阵。`ℓ_j = v_j^T A^{-1} v_j` 衡量的是 `v_j` 在**扣掉其余 value 张成的子空间之后还剩多少能量**：与别人共线时 `ℓ_j` 小，独占一个方向时接近 `|v_j|^2`。所以 `w_j * ℓ_j` 是「query 相关 × 信息不冗余」。

注意 A 里用的是 `√(w+eps)·v`，而二次型用的是**原始** `v`：上游把 `values_rep` 当 output-direction proxy（`capacity_vectors = values_rep`），缩放只进 A、不进 `ℓ`。两处不对称是设计，不是笔误。

## 计算流程（对应上游 `cap_press.py`）

1. `n_sink/n_future_positions/epsilon/tau` 分别是 `4 / 512 / 1e-6 / 5.0`（L62-67）；query 侧也先切掉 sink（L84）。
2. 平均旋转：对 512 个未来位置建 `cos·I + sin·rot` 再 `mean(dim=0)`（L136-137）。
3. `matmul(query_states, rotation.T)`（L162），再对 query 轴 `mean` 得 anchor（L175）——与逐 token 旋转再平均等价。
4. anchor 与 key 各自 `F.normalize`（默认 eps=1e-12）、`einsum` 得余弦、`clamp(-1, 1)`（L200-204）。
5. `logits = tau * statistic`，减 `amax` 后 `exp`（L219-221）。
6. `sqrt(weights + epsilon)` 缩放 value（L270），`einsum` 累加出 `A`（L279-283）。
7. `A` 转 float32 后 `linalg.solve`，`leverage = (v^T · A^{-1}v^T)` 按列求和，`scores = weights * leverage`（L285-291）。
8. `view(..., num_kv_groups, ...).mean(dim=2)` 回落到 kv head（L294-295）；`F.pad(..., n_sink, value=scores.max().item() + 1)`（L300）。

## 与上游的三处差异（题面已钉死）

1. **epsilon 的位置**：上游加在 `sqrt(weights + epsilon)` 里（L270），不是在 `A` 的对角线上。`A = I + 半正定项` 本身就正定，所以不需要抖动；测试里 `v=0` 的列必须给出 `ℓ = 0`，加了对角抖动就变成 `1/eps`。
2. **内部精度**：上游把 `A` 与 value 都 `.to(torch.float32)` 再 solve（L285/287）。本题按仓库规则内部 float64、输出回输入 dtype；float32 用例的容差是 3e-5。
3. **入参形态**：上游从 `hidden_states` 现算 q/k/v，并在内部生成 512 行 RoPE 表；本题把 q、`(cos, sin)` 表直接当入参（表长任取，只取均值），`keep` 显式给而不是由 `compression_ratio` 反推，同分取更小下标（上游 `topk` 不定义）。

## 与 `kv-expected-attention` 的分工

两题共享同一副骨架：切掉 `n_sink=4`、平均 512 个未来位置的 RoPE、sink 填**全局 max + 1**。差别在分数的物理含义：

- **EA** 站在 query 侧：用高斯 query 模型算 `E[e^{q^Tk/√d}]`，softmax 后乘 `||v||`，问的是「未来会给这条 key 多少注意力、value 有多大」；
- **CapKV** 站在 value 侧：query 只用来生成权重 `w_j`，主项是 `v_j` 在容量矩阵下的 leverage，问的是「这条 value 有多少信息是别人解释不了的」。

所以 EA 是逐元素打分，CapKV 每个 head 要做一次 `D×D` 的 solve 并且对 GQA 组取均值。EA 那道题要求自己算 `mu/cov`；本题的 anchor 只用到 mean-then-rotate，没有协方差项。

## 判分点与边界

- `statistic` 必在 `[-1, 1]`；`weights` 每个 head 的最大值恰为 1（max-shift 的结果）。
- `k_j = 0` 或 anchor 为 0 时余弦必须是 0（normalize 的 eps=1e-12），不能是 NaN。
- `v_j = 0` 时 `ℓ_j = 0`、分数 0；`v` 全零时所有真实分数为 0，`kept` 就是 `0..keep-1`（sink + 最小下标）。
- sink 分数 = **跨 head** 的全局 max + 1，测试专门造了一个「head 1 分数全 0」的用例。
- `kept` 每行升序、无重复、前 4 位恒为 `0..3`。
