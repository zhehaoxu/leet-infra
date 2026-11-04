# 扩散：MagCache 累积误差门控步（retention 保留 + 累积误差跳过）

## 题意

MagCache（Magnitude-aware Cache）让扩散 Transformer 在去噪后期**跳过整段块**：与 FBCache（764）逐步比较残差不同，它维护**累积误差**——连续跳步时幅度比连乘、误差累加，直到超过阈值才恢复执行。实现 `solve(hidden_states, tail_output, state, step, num_inference_steps, mag_ratios, threshold=0.06, retention_ratio=0.2, max_skip_steps=3)`：

```text
retention_step = int(retention_ratio * num_inference_steps + 0.5)   # 四舍五入
step < retention_step  ->  强制 compute（不累积）

step >= retention_step（先累积、后判定）:
    ratio *= mag_ratios[step]          # 越界回退 1.0
    steps += 1
    err   += abs(1.0 - ratio)          # 累积 |1-Πscale|，非单步 |1-scale|
    skip   <=>  previous_residual is not None
                and err   <= threshold
                and steps <= max_skip_steps
    不满足 -> 重置 (ratio=1.0, steps=0, err=0.0) 并 compute
```

- **compute 路径**：`output = tail_output`（完整执行结果）；`previous_residual = tail_output − hidden_states`。
- **skip 路径**：`output = hidden_states + previous_residual`（上次 compute 的残差），累积器**保留不重置**。

返回 `dict(output, computed, accumulated_err, state)`，state 为新 dict（纯函数式），键 `{previous_residual, accumulated_ratio, accumulated_steps, accumulated_err}`。

`mag_ratios` 长度 ≠ `num_inference_steps` 时用官方 `nearest_interp` 最近邻插值：`scale = (src_len−1)/(target_len−1)`、`idx = round(grid·scale)`（torch.round 银行家舍入）、target==1 取末元素。

## 关键语义

- **先累积后判定**：err 的累积发生在判定**之前**——即使本次必须 compute（如首步残差为 None），ratio/steps/err 也已加上本步贡献；只有走 compute 分支才重置。skip 步累积器保留（连续 skip 时 ratio 继续连乘、err 继续累加）。
- **err 是 |1−连乘积| 的累加**，不是 |1−scale| 的累加：0.98 连乘两步得 err = 0.02 + |1−0.98²| = 0.0596，不是 0.04。
- **retention 四舍五入**：`int(ratio·N + 0.5)`——0.2·10=2.0→2、0.25·10=2.5→3、0.2·8=1.6→2；截断版 `int(ratio·N)` 在 frac(ratio·N)≥0.5 时结果不同。
- **stale 残差**：skip 步输出用的是**上次 compute 步**的残差，与本次 tail_output 无关。
- **scale 越界回退 1.0**（官方同款）：step ≥ len(ratios) 时 scale=1.0，ratio 不变但 err 仍按 |1−ratio| 累积。

## 判题域约定

- 官方 hook 挂 head/tail 块共享 state，step_index 由尾块 hook 推进；判题域 step 显式入参。
- 官方 calibrate 模式（测 ratios 用）不入实现；mag_ratios 必须显式提供。
- 官方 skip 时文本+图像拼接分支（Flux/SD3 形状不匹配专用）不入实现；判题域要求 hidden_states 与 tail_output 形状一致。
- state 四键白名单校验（未知键先于缺键报错）、`threshold ≥ 0`、`retention_ratio ∈ [0,1]`、`max_skip_steps ≥ 1` 为判题域显式校验。

## 例子

N=10、retention_ratio=0.2 → retention_step=2：步 0/1 强制 compute。步 2 起 ratios=[0.98,…]：步 2 残差为 None → compute+重置；步 3 ratio=0.98、err=0.02≤0.06 → skip（output = hidden + 旧残差）；步 4 err=0.0596 仍 skip；步 5 err=0.1184>0.06 → compute+重置。

数据规模：张量 `(1, 2)`–`(2, 2)` float64，N ∈ [1, 10]，threshold ∈ [0, 100]。
