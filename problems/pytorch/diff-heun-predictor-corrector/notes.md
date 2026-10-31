## 上游

- 仓库：`huggingface/diffusers`，`src/diffusers/schedulers/scheduling_heun_discrete.py`（v0.42.x，2026-10-31 经 contents API 下载；本地副本 `.workbuddy/upstream/d93/`）

## 逐行对照

| 上游 | 本题 |
| --- | --- |
| `set_timesteps` L348-377 | sigma 布局：`self.sigmas = cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])` → 长 2N、末位 0；`timesteps = cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])` → 长 **2N−1**（比 sigmas 少一——最后一个逻辑步只有预测，校正半步永远遇不到 σ=0） |
| `state_in_first_order` L538-540 | `state is None` 分支（上游判 `self.dt is None`） |
| `step` L588-600 σ 选择 | 预测：`sigma = sigmas[i]`、`sigma_next = sigmas[i+1]`；校正：`sigma = sigmas[i-1]`、`sigma_next = sigmas[i]`（数值上与预测半步同一对 (s_j, s_{j+1})） |
| `step` L602-616 x0 分支 | `sigma_input = sigma_hat`（预测）vs `sigma_next`（校正）；v_prediction 公式逐字保留 |
| `step` L623-648 预测 | `derivative = (sample − x0)/sigma`、`dt = sigma_next − sigma`、暂存 `prev_derivative/dt/sample`、`prev = sample + derivative·dt` |
| `step` L633-646 校正 | `derivative = (sample − x0)/sigma_next`、`avg = (prev_derivative + derivative)/2`、`prev = self.sample + avg·self.dt`（**暂存**样本，非当前），随后清空状态 |
| `step` L618-621 clip | `clip_sample` 在导数之前对 x0 clamp |

## 刻意偏离（已记录）

- **不依赖 diffusers**：判题环境无 diffusers；`sigmas`（已展开成 2N 成对布局）为显式入参。`set_timesteps` 的调度构造（linspace/leading/trailing、Karras/exponential/beta）不在范围。
- **状态显式传递**：上游把状态挂在 `self`（`prev_derivative`/`dt`/`sample` 属性）；本题预测半步返回 state dict、校正半步传入——信息相同，但可作纯函数判题。
- **补充结构校验**：奇数 i 的 `sigmas[i] == sigmas[i+1]`、预测索引偶/校正索引奇、校正索引 ≤ 2N−3、相应半步 σ > 0。上游都不校验（靠 pipeline 约定）；这些编码了 pipeline 对 2N−1 个 timesteps 循环调用的真实模式。
- **`gamma = 0` 固定**（L596-600）：上游支持 `sigma_hat = sigma·(gamma+1)` 但硬编码 gamma=0，不参数化。
- 内部 float64 计算；输出返回 float32。

## 测试设计

- 独立 numpy oracle `_oracle_half` 从 step 方程重写；N=3 的完整 5 次调用轨迹逐元素对拍（1e-9），含末步「只有预测」（dt = 0 − s_{N-1}）。
- 手算示例带硬编码锚点：预测 x0=[−6,−1]、euler=[−2,−1]；校正 avg=[0.5,0.5]、prev=[0,−3]（从**暂存** sample [2,−1] 出发）。
- 反例已写入 description：从当前 sample 出发得 [−4,−3]——恰好差一个欧拉半步。
- v_prediction 分支 + clip_sample 在导数之前（mo=0、x0=sample 被 clamp 到 ±0.5 → derivative = ±(10−0.5)/4）。
- 9 条校验用例（形状、成对布局、索引奇偶、校正越界、state 键）。
- 变异测试 5/5 抓获：校正从当前 sample 出发、校正除以 σ_next+1、预测除以 σ_next、不取平均、dt 重算而非复用。
