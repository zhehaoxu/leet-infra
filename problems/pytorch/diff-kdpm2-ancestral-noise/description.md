## 题意

确定性 KDPM2（754 题）把 DPM-Solver-2 摊在交错 σ 表上但**不加噪声**；ancestral 版（`KDPM2AncestralDiscreteScheduler`）在同一个两调用骨架上做三件事：① 中点不再插在 σ 与 σ_next 之间，而是插在 σ 与 **σ_down** 之间；② 次调的 dt 从「整步长 σ_next − σ」改成「**σ_down − σ**」；③ prev 落地后**注入 `noise · σ_up`**。σ_up² + σ_down² = σ_next²——ancestral 的语义就是把确定性步长拆成「确定性部分 σ→σ_down」与「噪声部分 σ_up」，噪声量级由 σ 比值决定。

判题输入：

- `model_output` / `sample`：同形状浮点张量（次调的 `sample` 是**中点样本**）；;
- `sigmas`：一维浮点张量，**未展开**的基础表 `[σ0, …, σ_{N−1}, 0]`（长度 N+1，N ≥ 2），有限、非负、单调不增，前 N 个 > 0、末位恰为 0；
- `step_index`：当前索引；
- `state`：`None` = 首调；`{"sample"}` = 次调（首调返回值原样传入）；
- `noise`：**次调必需**（官方由 generator 生成，本题改显式入参），形状须与 `model_output` 一致；首调忽略；
- `prediction_type` ∈ {`epsilon`, `v_prediction`}。

## 计算规则

**辅助表的构造**（每次调用重算，官方 `set_timesteps` 逐字同构）：

- `σ_next = roll(−1)`（`σ_next[N] = 0`）；
- `up[k] = √( σ_next[k]²·(σ[k]² − σ_next[k]²) / σ[k]² )`——k=N 处 0/0 = **nan**（官方同款，该格不被读）；
- `down[k] = √( σ_next[k]² − up[k]² )`，随后**强制 `down[N] = 0`**（清 k=N 处的 nan）；
- `interpol[k] = exp( lerp(log σ[k], log down[k], 0.5) )`，随后**强制 `interpol[N−1] = interpol[N] = 0`**——torch 路径实测 `lerp(有限, −inf, 0.5) = nan`（不是 −inf！），k=N−1（`down[N−1]=0 → log = −inf`）与 k=N（两端都 −inf）**两格都是 nan**，官方 `[-2:] = 0.0` 清场一格都不能少；
- 数学恒等式：`down[k] = σ_next[k]²/σ[k]`、`√(σ[k]·down[k]) = σ_next[k]`——**interpol 表精确数学下与 σ_next 逐元素相等**，官方显式构造是冗余的（见 notes 的消去讨论）；
- 四张展开表（长度 **2N+2**，头 1 条 + 主体 repeat_interleave(2) + 尾 1 条）：`sigE / sigIE / sigUp / sigDown` 同构。**注意与 754 不同**：ancestral 的 interpol 表没有 `[0,…]` 死头——`interpol[0] = √(σ0·down0)` 本身就是逻辑步 0 的中点。

**索引结构**：逻辑步 j（0 ≤ j ≤ N−1）占索引 2j（首调）与 2j+1（次调）；**末逻辑步只有首调**（次调 i ≤ 2N−3）。首调与次调读**同一格**中点：`sigIE[2j] = interpol[j]`。

**首调**（`state is None`，i 偶数，0 ≤ i ≤ 2N−2）：

- `σ = sigE[i]`（σ_hat = σ·(gamma+1)，gamma = 0 恒等）、`m = sigIE[i]`；
- `x0 = pred(sample, mo, σ)`；`derivative = (sample − x0)/σ`（分母是 σ）；
- `dt = m − σ`（半步到中点）；`prev = sample + derivative·dt`；
- `state = {"sample": sample}`——只存**原样本**；`noise` 被忽略；返回 `sigma_up / sigma_down` 为 `None`。官方首调还取出 `sigmas_up[i]`、`sigmas_down[i−1]`，但首调分支不读它们（死赋值，i=0 时 `down[−1]` 走负索引也不炸）。

**次调**（state 为只含 `sample` 的 dict，i 奇数，1 ≤ i ≤ 2N−3）：

- `σ = sigE[i−1]`、`m = sigIE[i−1]`、`u = sigUp[i−1]`、`d = sigDown[i−1]`；
- `x0 = pred(sample, mo, m)`；`derivative = (sample − x0)/m`（分母是**中点**）；
- `dt = d − σ`——**不是整步长**：噪声吃掉 up 部分；
- `prev = state["sample"] + derivative·dt + noise·u`——从**暂存样本**出发并注入噪声；`state = None`；首调导数被丢弃。

**prediction_type**（sigma_input 首调用 σ、次调用 m）：

- `epsilon`：`x0 = sample − sigma_input·mo`；
- `v_prediction`：`x0 = mo·(−sigma_input/√(sigma_input²+1)) + sample/(sigma_input²+1)`。

末逻辑步 j=N−1：σ_next = 0 → up = down = interpol = 0，一步落到 x₀ 且**无噪声**。

返回 dict：`prev_sample` / `pred_original_sample`（float32）、`state`（dict 或 None）、`derivative`（float32）、`dt`（浮点数）、`sigma_up` / `sigma_down`（次调的浮点值，首调为 None）。校验失败一律抛 `ValueError`。

## 示例

```
input : sigmas = [4, 1, 0], first call mo = 0.3, sample = 2.0, i = 0
output: down0 = 1^2/4 = 0.25, interpol[0] = sqrt(4*0.25) = 1.0
        x0 = 2 - 4*0.3 = 0.8, derivative = (2 - 0.8)/4 = 0.3
        dt = 1.0 - 4 = -3.0, prev = 2 + 0.3*(-3) = 1.1
        state = {"sample": 2.0}, sigma_up / sigma_down = None

input : second call mo = -0.5, sample = 1.1 (midpoint), noise = 1.5, i = 1
output: up0 = sqrt(1*(16-1)/16) = sqrt(0.9375), down0 = 0.25
        x0 = 1.1 - 1.0*(-0.5) = 1.6, derivative = (1.1 - 1.6)/1.0 = -0.5
        dt = 0.25 - 4 = -3.75
        prev = 2.0 + (-0.5)*(-3.75) + 1.5*sqrt(0.9375) = 3.875 + 1.4524 = 5.3274
        state = None, sigma_up = sqrt(0.9375), sigma_down = 0.25
```
