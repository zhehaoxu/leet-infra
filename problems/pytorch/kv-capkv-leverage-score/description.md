# KV Cache：CapKV 的信息论 leverage score token 选择

## 题目描述

KV cache 压缩通常拿注意力分数当判决书。CapKV（arXiv 2604.25975）换了个判据：如果某个 token 的 value 能被**其余 value 的线性组合**解释，它就是冗余的，保留它只是在重复已有信息；衡量「不好解释」程度的量是 value 矩阵的 **leverage score**。分数分两步算。

**第一步：用未来的 RoPE 造一个 query anchor。** 真正的打分发生在后面若干 decode step，所以先把未来 `n_future_positions=512` 个位置的旋转矩阵平均掉，再作用到 query 的均值上（本题把 `(cos, sin)` 表直接当入参，表长任取）：

\[
\bar q=\Bigl(\tfrac{1}{L_q}\sum_i q_i\Bigr)\bar R^{\top},\qquad
\bar R=\tfrac{1}{P}\sum_{p=1}^{P}R_p,\qquad R_p=\operatorname{diag}(\cos_p)+\sin_p\cdot \mathrm{rot}
\]

逐 token 旋转再平均、与先平均再旋转完全等价（矩阵乘是线性的），上游按后者写。

**第二步：加权 leverage score。** 先算每个 key 与 anchor 的余弦统计量，再把它变成指数权重：

\[
s_j=\operatorname{clamp}\bigl(\cos(\bar q,k_j)\bigr),\qquad
w_j=\exp\bigl(\tau(s_j-\max_{j'}s_{j'})\bigr),\quad \tau=5
\]

然后以 value 为列拼容量矩阵，对每条 value 求它在该矩阵逆下的二次型：

\[
A=I+\sum_j (\sqrt{w_j+\epsilon}\,v_j)(\sqrt{w_j+\epsilon}\,v_j)^{\top},\qquad
\ell_j=v_j^{\top}A^{-1}v_j
\]

最终分数是 `w_j * ℓ_j`：权重高且真的贡献了独立方向的 token 分数高；权重高但 value 与别人共线时 `ℓ_j` 会很小。三个容易写错的细节：

1. **前 4 个 token 是 sink**，既不参与 statistic/权重/`A`，也不进 leverage——但**永远保留**，分数直接填全局最大值 + 1；
2. **GQA 组内取均值**：`repeat_kv` 把同一个 kv head 连续铺 `group_size` 份，所以按 `group_size` 切块求均值，不是取第一份；
3. `epsilon=1e-6` 加在 `sqrt(w+epsilon)` 里面（给权重做正则，保证 `A` 严格正定），**不是**加在 `A` 的对角线上。

## 任务

实现 `solve(q, k, v, cos, sin, keep, group_size=1)`：

- `q`：`(Lq, Hq, D)`，`Hq = Hkv * group_size`；`k`、`v`：`(T, Hkv, D)` 同形状，与 `q` 同为 float32/float64；`cos`、`sin`：`(P, D)` 同形状的 RoPE 表，`P >= 1`，`D` 是正偶数；`keep`：`[4, T]` 内的整数，`group_size` 为正整数。

内部一律 float64，返回按输入 dtype（`kept` 除外），且不允许原地改输入。返回 5 键字典，记 `T' = T - 4`：

- `statistic`、`weights`、`leverage`：`(Hq, T')`，只覆盖 sink 之后的 token；
- `scores`：`(Hkv, T)`，头部 4 列是 sink 的 `scores.max() + 1`（全局 max，跨 head）；
- `kept`：`(Hkv, keep)` int64，每行按分数降序取 `keep` 个再升序输出，同分取更小的 token 下标。

非法输入抛 `ValueError`：非三维、dtype 不一致或不是 float32/float64、`k`/`v` 形状不同、`D` 不是正偶数、`Lq < 1`、`group_size` 不是正整数、`Hq != Hkv * group_size`、`T <= 4`、`cos`/`sin` 不是同形状的 `(P, D)`、`keep` 不是 `[4, T]` 内的整数。

## 示例

```
q  = [[[1.0, 0.0]], [[0.0, 1.0]]]                        # (Lq=2, Hq=1, D=2)
k  = [[[9.0, 9.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]]    # (T=6, Hkv=1, D=2)
v  = [[[0.0, 0.0]]] * 4 + [[[1.0, 0.0]], [[0.0, 1.0]]]
cos = [[1.0, 1.0], [0.0, 0.0]]                           # P=2 个未来位置
sin = [[0.0, 0.0], [1.0, 1.0]]
keep = 5

mean(R)   = (I + rot) / 2 = [[0.5, -0.5], [0.5, 0.5]]    # mean(q) = [0.5, 0.5]，rot = [[0, -1], [1, 0]]
anchor    = [0.5, 0.5] @ mean(R)^T = [0.0, 0.5] -> normalize -> [0.0, 1.0]
statistic = [[0.0, 1.0]]                                 # 与 e1 / e2 的余弦
weights   = [[0.0067379, 1.0]]                           # [exp(-5), 1]
leverage  = [[0.9933062, 0.49999975]]                    # 1 / (1 + w_i + eps)
scores    = [[1.49999975] * 4 + [0.0066928, 0.49999975]]  # w_i * leverage_i，前 4 位填 max + 1
kept      = [[0, 1, 2, 3, 5]]                            # 降序取 5 个后升序
```

## 约束

- **normalize 的 eps 是 1e-12**：`k_j = 0` 或 anchor 退化成 0 时余弦记 0，不是 NaN；统计量再 clamp 到 `[-1, 1]`。**max-shift 后权重上界恰为 1**：`exp(τ(s - max s))` 不会溢出，也让 sink 的 `max + 1` 一定压过所有真实分数。
- **`sqrt(w + eps)` 里的 `w` 是未缩放的权重**：容量矩阵用 `√(w+eps)·v` 拼，leverage 的二次型却用**原始** `v`；两处用错任一处，分数就不是 CapKV 的分数。
- **`A` 天然正定**：`A = I + 半正定项`，所以直接 `linalg.solve`，不要求逆、也不加对角抖动；`v_j = 0` 时 `ℓ_j = 0`，分数为 0，不会出 NaN。
- **sink 的填充值是全局 max + 1**：`scores.max()` 跨所有 kv head，而不是每个 head 自己的 max + 1；多 head 时两种写法在分数为 0 的 head 上结果不同。
- **同分取更小的 token 下标**：降序 `argsort` 必须是 stable 的，`torch.topk` 不定义同分顺序，本题把它钉死。
