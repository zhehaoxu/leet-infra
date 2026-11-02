## 题意

Heun（643 题）用「两个导数取平均」做二阶；KDPM2 走的是另一条路——Karras et al. 的 **DPM-Solver-2**：在 log-σ 的**几何平均中点**上重估一次导数，然后拿这一个导数从**步起点**直接走完整个步长，首调的导数直接丢弃。diffusers 的 `KDPM2DiscreteScheduler` 把它实现成一张**交错 σ 表**上的两调用状态机：基础 σ 表先经 log-空间 lerp 造出相邻 σ 的中点表，再把两张表各自 repeat_interleave 成长度 2N+2 的展开表，每个逻辑步占两个 step 索引。

判题输入：

- `model_output` / `sample`：同形状浮点张量（次调的 `sample` 是**中点样本**，`model_output` 是中点上的模型输出）；
- `sigmas`：一维浮点张量，**未展开**的基础表 `[σ0, …, σ_{N−1}, 0]`（长度 N+1，N ≥ 2），有限、非负、单调不增，前 N 个 > 0、末位恰为 0；
- `step_index`：当前索引；
- `state`：`None` = 首调；`{"sample"}` = 次调（首调返回值原样传入）；
- `prediction_type` ∈ {`epsilon`, `v_prediction`}。

## 计算规则

**展开表的构造**（每次调用重算）：

- 中点表：`I[k] = √(σ[k−1]·σ[k])`（k = 1…N），`I[0] = 0`。与官方 `sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()` 数值同构（√(ab) = exp((log a + log b)/2)）；
- 两张展开表（长度 **2N+2**，头 1 条 + 主体 repeat_interleave(2) + 尾 1 条）：
  - `sigE = [σ0, σ1,σ1, σ2,σ2, …, σ_{N−1},σ_{N−1}, 0, 0, 0]`
  - `sigIE = [0, I1,I1, I2,I2, …, I_{N−1},I_{N−1}, 0, 0, 0]`

即 `sigE[2k] = σ_k`（0 ≤ k ≤ N）、`sigE[2k+1] = σ_{k+1}`（0 ≤ k ≤ N−1）、`sigE[2N+1] = 0`；`sigIE` 同构。

**索引结构**：逻辑步 j（0 ≤ j ≤ N−1）占索引 2j（首调）与 2j+1（次调）；**最后一个逻辑步只有首调**（timesteps 长 2N−1，末次调用在索引 2N−2，中点 `sigIE[2N−1] = 0`，dt = −σ_{N−1} 一步落到 x₀）。

**首调**（`state=None`，索引为偶数，0 ≤ i ≤ 2N−2；σ_hat = σ·(gamma+1)，官方 gamma=0 恒等）：

- `σ = sigE[i]`（须 > 0），中点 `m = sigIE[i+1]`；
- `x0 = f(sample, mo, σ)`，`derivative = (sample − x0) / σ`；
- `dt = m − σ`，`prev = sample + derivative·dt`（欧拉半步走到中点）；
- 暂存 `state = {"sample": sample}`——只存**原样本**。

**次调**（`state` 为 dict，索引为奇数，1 ≤ i ≤ 2N−3；末逻辑步没有次调）：

- `σ = sigE[i−1] = σ_j`，中点 `m = sigIE[i] = I_{j+1}`（须 > 0），`σ_next = sigE[i] = σ_{j+1}`；
- `x0 = f(sample, mo, m)`——sigma_input 用**中点**；
- `derivative = (sample − x0) / m`——分母是**中点**；
- `dt = σ_next − σ`——**整步长**，不是半步；
- `prev = state["sample"] + derivative·dt`——从**暂存样本**出发；首调导数被丢弃；
- 返回 `state = None`。

**x0 的两种口径**（sigma_input：首调取 σ、次调取中点 m）：

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_input·mo` |
| `v_prediction` | `mo·(−σ_input/√(σ_input²+1)) + sample/(σ_input²+1)` |

返回 `{"prev_sample", "pred_original_sample", "state", "derivative", "dt"}`。

## 手算示例

```
基础表 [4, 1, 0]（N=2）→ I1 = √(4·1) = 2
sigE  = [4, 1, 1, 0, 0, 0]；sigIE = [0, 2, 2, 0, 0, 0]
首调 i=0（σ=4, m=2）：sample=1, mo=0.1
  x0 = 0.6；d = 0.1；dt = 2−4 = −2；mid = 1 − 0.2 = 0.8；state 存 sample=1
次调 i=1（σ=4, m=2, next=1）：sample'=0.8, mo'=0.2
  x0 = 0.8 − 0.4 = 0.4；d2 = (0.8−0.4)/2 = 0.2；dt = 1−4 = −3（整步）
  prev = 1 + 0.2·(−3) = 0.4      ← 从暂存 sample=1 出发
首调 i=2（σ=1, m=0）：dt = −1；prev = 0.8 − 1·0.3 = 0.5（epsilon 即 x0）
```

**两个反例**：

- 次调若改从当前中点样本出发：`0.8 + 0.2·(−3) = 0.2`——与正确答案 0.4 差了整个半步位移；
- 次调导数若除以 σ_next=1 而不是中点 2：`d2 = 0.4`，`prev = 1 − 1.2 = −0.2`——直接跑飞。

## 边界与约束

- `sigmas` 长 ≥ 3、有限、非负、单调不增、末位恰为 0、前 N 个 > 0；
- 首调索引必须为偶数、次调必须为奇数；次调索引 ≤ 2N−3（末逻辑步只有首调）；首调索引 ≤ 2N−2；
- 首调 σ > 0、次调 σ > 0 且中点 m > 0；
- `state` 必须恰含 `sample` 一键且与 sample 同形状；`step_index` 为整数（bool 拒绝）；
- 内部 float64 计算，`prev_sample`/`pred_original_sample`/`derivative` 返回 float32；违例抛 `ValueError`。

## 与现有题的边界

- 643 `diff-heun-predictor-corrector`：同为「一逻辑步两次调用 + 暂存样本」的状态机，但 Heun 的中点是**算术半步**（σ_next 不变、走半程再回来平均两个导数）；KDPM2 的中点是 log-σ **几何平均**、只保留中点导数、dt 是整步长——三处机制差逐一对应三条不同的变异；
- 522 `diff-euler-ancestral-step`：σ 空间欧拉 + 祖先噪声拆分，单步无状态；
- 643 题的 sigmas 是**已展开**的成对表入参；本题要求解题者**自己构造**交错表（几何中点 + repeat_interleave 的头尾补齐），构造本身是判据的一部分；
- KDPM2 的 ancestral 变体（σ_up/σ_down 拆分 + 噪声注入）不在本题范围，机制上更接近 522；
- 494–513 训练侧与引导族、576 UniPC：机制无关。
