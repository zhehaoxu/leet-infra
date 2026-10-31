## 题意

Euler 离步（522 题的确定性内核）是一阶的；Heun 方法用「预测-校正」把同样的模型调用量升级成二阶：先用欧拉走半步探路，再在新点上重估导数，两个导数取平均后**从头**走完整个步长。diffusers 的 `HeunDiscreteScheduler` 把这套机制实现成一个两阶段状态机——每个逻辑噪声级占**两个** step 索引，中间状态（导数、步长、样本）靠 scheduler 自身暂存。

判题输入：

- `model_output` / `sample`：同形状浮点张量（校正半步的 `sample` 是**欧拉半步结果**，`model_output` 是新点上的模型输出）；
- `sigmas`：一维浮点张量，按上游 `set_timesteps` 的成对结构展开：`[s0, s1, s1, s2, s2, …, s_{N-1}, s_{N-1}, 0]`，长度 2N；
- `step_index`：当前索引；
- `state`：`None` = 预测半步；`{"prev_derivative", "dt", "sample"}` = 校正半步（预测半步的返回值原样传入）；
- `prediction_type` ∈ {`epsilon`, `v_prediction`, `sample`}、`clip_sample`、`clip_sample_range`。

## 计算规则

**索引结构**：逻辑步 j（0 ≤ j ≤ N−2）占索引 2j（预测）与 2j+1（校正）；**最后一个逻辑步只有预测**（索引 2N−2，σ_next = 0）。timesteps 长 2N−1、sigmas 长 2N——正因 sigmas 比 timesteps 多一位，校正半步永远不会遇到 σ=0。

**预测半步**（`state=None`，索引为偶数）：

- `σ = sigmas[i]`（须 > 0），`σ_next = sigmas[i+1]`；
- `x0 = f(sample, mo, σ)`，`derivative = (sample − x0) / σ`；
- `dt = σ_next − σ`，`prev = sample + derivative·dt`；
- 暂存 `state = {"prev_derivative": derivative, "dt": dt, "sample": sample}`。

**校正半步**（`state` 为 dict，索引为奇数且 ≤ 2N−3）：

- 要求 `sigmas[i] == sigmas[i+1]`（成对结构）；`σ_next = sigmas[i]`（须 > 0）；
- `x0 = f(sample, mo, σ_next)`，`derivative = (sample − x0) / σ_next`；
- `avg = (state.prev_derivative + derivative) / 2`；
- `prev = state.sample + avg·state.dt`——**从暂存样本出发、用暂存步长**，不是从当前 sample 出发；
- 返回 `state = None`（清空）。

**x0 的三种口径**（σ_input：预测半步取 σ、校正半步取 σ_next）：

| prediction_type | x0 |
| --- | --- |
| `epsilon` | `sample − σ_input·mo` |
| `v_prediction` | `mo·(−σ_input/√(σ_input²+1)) + sample/(σ_input²+1)` |
| `sample` | `mo` |

`clip_sample=True` 时对 x0 clamp 到 ±`clip_sample_range`（发生在导数计算**之前**）。

返回 `{"prev_sample", "pred_original_sample", "state", "derivative"}`。

## 手算示例

```
sigmas = [8, 4, 4, 1, 1, 0]（N=3），epsilon
预测 step0（σ=8, next=4）：sample=[2,−1], mo=[1,0]
  x0 = [−6, −1]；d = [1, 0]；dt = −4；euler = [−2, −1]
校正 step1（σ_next=4）：sample'=[−2,−1], mo'=[0,1]
  x0' = [−2, −5]；d2 = [0, 1]；avg = [0.5, 0.5]
  prev = [2,−1] + (−4)·[0.5,0.5] = [0, −3]   ← 从暂存 sample=[2,−1] 出发
```

**从当前 sample 出发的反例**：同一校正半步若改从 `sample'=[−2,−1]` 出发，得 `[−2,−1] + (−4)·[0.5,0.5] = [−4,−3]`——与正确答案差了整个欧拉半步。

**导数分母的反例**：预测半步的导数除以**当前 σ**，校正半步除以 **σ_next**；把两处写成同一个 σ，在 σ_next ≠ σ 的成对边界（如末逻辑步预测 σ_next=0）直接除零。

## 边界与约束

- `sigmas` 长 ≥ 3、有限、非负、单调不增；成对结构由 `sigmas[i] == sigmas[i+1]`（i 奇数）校验；
- 预测半步索引必须为偶数、校正半步必须为奇数，且校正索引 ≤ 2N−3（末逻辑步只有预测）；
- 预测半步 σ > 0、校正半步 σ_next > 0；
- `state` 必须恰含三键且张量与 sample 同形状，`dt` 为数；
- 内部 float64 计算，`prev_sample`/`pred_original_sample`/`derivative` 返回 float32；违例抛 `ValueError`。

## 与现有题的边界

- 522 `diff-euler-ancestral-step`：同是 σ 空间欧拉，但带祖先噪声拆分与随机注入（单步、无状态）；本题是确定性两阶段状态机、无噪声；
- 519 `diff-edm-preconditioning`：EDM 的 c_skip/c_out/c_in 预条件网络封装，不涉及多调用状态机；
- 576 `diff-unipc-pc-step`：UniPC 的统一预测-校正也有「校正」二字，但系数来自多变量多项式拟合且无成对 σ 展开；
- 494–496 等 SD3 训练侧与 501–513 引导族：机制无关。
