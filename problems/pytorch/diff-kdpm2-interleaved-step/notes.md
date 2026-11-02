# notes

## 上游

- 仓库：`huggingface/diffusers`，`src/diffusers/schedulers/scheduling_k_dpm_2_discrete.py`（715 行，2026-11-02 经 contents API + `Accept: application/vnd.github.raw` 下载）。ancestral 版 `scheduling_k_dpm_2_ancestral_discrete.py`（750 行）一并对照，确认 up/down 拆分不在本题范围。

## 逐行对照

| 上游 | 本题 |
| --- | --- |
| `set_timesteps` L323-327 | 基础表末位补 0 后，`sigmas_interpol = sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()`——log-空间与 `roll(1)`（前一位）的插值 = 相邻 σ 的几何平均。本题用 `√(σ[k−1]·σ[k])` 同构实现（`√(ab) = exp((log a + log b)/2)`，仅差末位 ulp，oracle 自洽） |
| `set_timesteps` L329-332 | 展开表：`cat([t[:1], t[1:].repeat_interleave(2), t[-1:]])`，长度 2N+2。结构 `sigE[2k]=σ_k`、`sigE[2k+1]=σ_{k+1}`、尾部三个 0 |
| `set_timesteps` L343-345 | `interleaved_timesteps = stack((timesteps_interpol[1:-1], timesteps[1:])).flatten()`——timesteps 长 **2N−1**，比 sigmas 少一位 → **末逻辑步只有首调**（与 643 题同构的「sigmas 比 timesteps 多一位」论证） |
| `step` L601-615 σ 选择 | 首调：`sigma = sigE[i]`、`sigma_interpol = sigIE[i+1]`、`sigma_next = sigE[i+1]`（取出但不用）；次调：`sigma = sigE[i−1]`、`sigma_interpol = sigIE[i]`、`sigma_next = sigE[i]` |
| `step` L614-615 | `gamma = 0` 硬编码，`sigma_hat = sigma·(gamma+1)` 恒等；本题不参数化 |
| `step` L618-631 x0 | epsilon / v_prediction 两支逐字保留；`sample` 类型抛 NotImplementedError → 本题归入未知类型抛 ValueError |
| `step` L633-640 首调 | `derivative = (sample − x0)/sigma_hat`、`dt = sigma_interpol − sigma_hat`、`self.sample = sample`（**只存样本**，不存 dt/导数） |
| `step` L641-650 次调 | 注释就叫 `# DPM-Solver-2`：`derivative = (sample − x0)/sigma_interpol`、`dt = sigma_next − sigma_hat`（**整步长**）、`sample = self.sample`（回滚）后清空 |
| `step` L655 | `prev_sample = sample + derivative * dt` 两个分支共用 |

## 官方怪癖（本题如实保留/记录）

1. **`I[0]` 从不被读**：torch 路径下 `log(0) = −inf` 参与 lerp，`I[0] = exp(nan) = nan`、`I[N] = exp(−inf) = 0`；numpy 路径反之（`I[0]=0`、`I[N]=nan`）。展开后 nan 落在 `sigIE[0]`（torch）——step 的所有读取索引 ≥ 1，**从不消费**。ancestral 版干脆加了 `sigmas_interpol[-2:] = 0.0`（L337）清场，是作者撞过这坑的直接证据。本题的几何均值构造天然得 0，行为与 torch 路径在所有被读索引上一致。
2. **末逻辑步一步到零**：`sigIE[2N−1] = I_N = 0`，末次首调 `dt = −σ_{N−1}`，epsilon 下 `prev == x0`（浮点下差 1 ulp 内）。
3. **首调的 `sigma_next = sigE[i+1]` 取出即弃**——这是 m-b-next-idx 等价性的根源（见下）。

## 与 643 Heun 的三点机制差（对应三条变异）

| | Heun (643) | KDPM2 (本题) |
| --- | --- | --- |
| 中点 | 算术半步（σ_next 复用、无中点表） | log-σ 几何平均 `√(σ_j·σ_{j+1})` |
| 导数合成 | `(d_预测 + d_校正)/2` | 只用中点导数，首调导数丢弃 |
| 次调 dt | 暂存的半步 dt | `σ_{j+1} − σ_j` 整步长 |
| 次调导数分母 | σ_next | 中点 m |
| state | 三键（导数+dt+样本） | 一键（样本） |

## 刻意偏离（已记录）

- **不依赖 diffusers**：基础 σ 表（未展开）为显式入参，展开表在 solve 内每次重算（纯函数、无 `self` 状态，state dict 显式传递）。
- **补充结构校验**：末位恰为 0、前 N 个 > 0、首调偶/次调奇、次调 ≤ 2N−3、首调 ≤ 2N−2、中点 > 0。上游不校验（靠 pipeline 约定）；这些编码了「pipeline 对 2N−1 个 timesteps 循环调用」的真实模式。
- 几何均值用 `sqrt` 直算而非 log-lerp：数值上仅末位 ulp 差，且避开 torch/numpy 在 −inf 上的路径分歧（见怪癖 1）。

## 测试设计

- 独立 numpy oracle `_oracle` 从 set_timesteps L323-345 + step L598-663 重写（含展开表构造）；13 个用例双模式 ACCEPTED（run 2/2 + submit 13/13）。
- 解析锚点用例：`[4,1,0]` 全轨迹 0.8 → 0.4 → 0.5（手算硬编码，非 oracle 对拍）；几何中点非完全平方 `√(3·2)`；`[2,2,0]` 退化（dt 全 0）；v_prediction 两阶手算。
- 随机对拍 6 组（N ∈ 2..4、随机 σ/mo、固定种子）逐调用对拍 prev/x0/dt；容差 1e-6——**solve 走 float32 往返**（pipeline 语义），oracle 是 float64，1e-9 会误报。
- 坑：末次调用（首调）返回 `state = {"sample": ...}` **不是 None**——官方 `self.sample` 不在末尾清空；测试最初断言末尾 state 为 None 是错的。
- 坑：测试曾把 oracle 的 numpy state 传回 solve（应传 `out["state"]` 的 torch 张量）→ `state.sample 必须是 torch 张量`。

## 变异测试

35 条变异：**30 CAUGHT + 5 等价**（等价实证后剔除口径记录如下）。

等价 5 条：

1. `m-geom-shift`：`(v[:-1]*v[1:])` → `(v[1:]*cat([v[:1], v[1:-1]]))`——逐元素同式（k=0 项 `s1·s0`，k≥1 项 `s_{k+1}·s_k`），200 组随机 σ `torch.equal` 全 0 差异；
2. `m-head-not-zero`：中点表头 `[0]` → `[I1]`——`I[0]` 只落 `sigIE[0]`，step 的读取索引全 ≥ 1，从不消费（**正好复刻官方怪癖 1**：官方 torch 路径此处本来就是 nan）；
3. `m-no-tail`：展开表去掉尾部补条（长度 2N+1）——`sigE[2N]`/`sigE[2N+1]` 无任何读取点（B 的最大读取索引 2N−3，A 的 2N−2）；
4. `m-ie-uses-sig-e`：`sigIE` 尾部用 `values[-1:]`（=0）替代 `interpol[-1:]`（=I_N=√(σ_{N−1}·0)=0）——两者恒等；
5. `m-b-next-idx`：次调 `sigma_next = sigE[i]` → `sigE[i+1]`——成对结构下 `sigE[2j+1] = σ_{j+1} = sigE[2j+2]`，对全部可达奇数 i（1..2N−3）恒等，200 组随机验证 0 差异。

抓取关键（首轮 SURVIVED 后补强）：

- `m-parity-b`（去掉次调奇偶检查）：原用例的偶数次调索引都落在范围外、被范围检查兜住**空抓**——补「N=3、i=2、带 state」用例（变异后静默算出错值）；
- `m-a-range-loose`（上界 2N−2 → 2N）：原用例 i=5 在 N=2 下仍越界空抓——补「i 恰为 2N」用例（变异后 σ=0 除零得 nan 不抛错）；
- `m-b-range-loose`（上界 2N−3 → 2N−1）：变异后末逻辑步次调滑到「中点为正」检查仍抛 ValueError 空抓——改为**断言错误消息**含 `step_index 必须落在`。
