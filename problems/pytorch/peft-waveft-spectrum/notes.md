# 详解：WaveFT 稀疏频谱增量权重

## 出处

peft `tuners/waveft.py`（WaveFT）。`WaveFTLayer.update_layer` 只做三件事：校验 \(n_\text{frequency}\)、按种子抽坐标、初始化频谱参数；真正的增量在 `get_delta_weight` 里按两条分支生成：

- **direct 路径**（`use_idwt=False`）：把频谱直接 scatter 进一个 \((d_\text{out},d_\text{in})\) 的稠密矩阵，乘 `scaling`；
- **IDWT 路径**（`use_idwt=True`，peft 默认）：先按小波族的尺寸缩减量 \((r_H,r_W)\) 把矩阵 pad 到 `d + reduction`、必要时补成偶数，再把原坐标整体平移 `(pad-d)//2` 居中，切成 `cA/(cH,cV,cD)` 四个子带后交给 `waverec2d` 重建，最后按中心裁剪回 \((d_\text{out},d_\text{in})\)，并同样乘 `scaling`。

本题实现 direct 路径。IDWT 路径依赖 `waverec2d`（移植自 PyTorch-Wavelet-Toolbox），不在判题环境的依赖里，因此明确切出范围——这也是 peft 中"更快但效果略差"的那条配置。

## 坐标语义（易错点）

1. **先扁平再还原**：`randperm(d_out*d_in)` 给出的是行优先展开的下标，落位用 `idx // d_in` 与 `idx % d_in`。写成 `divmod` 时把 `d_in` 换成 `d_out` 会在非方阵上错位。
2. **生成器必须带种子**：`torch.randperm(n, generation=generator)` 才有 peft 的可复现性；不传 generator 会吃全局 RNG 状态，同种子在不同调用顺序下结果不同。
3. **`scaling` 在增量侧**：`delta = dense * scaling`，不是乘在频谱上再 scatter（数值等价，但前向里 peft 就是把 `get_delta_weight` 的返回值直接加到基座权重上）。
4. **满预算不是一个 `reshape`**：\(n_\text{frequency}=d_\text{out}d_\text{in}\) 时 `randperm` 得到的是一个**置换**，所以 `delta.flatten()` 只是频谱的乱序，不是 `spectrum.reshape(d_out,d_in)`。`check_full_budget_covers_every_cell` 用排序对拍钉死这一点。

## 前向与合并

`WaveFTLinear.forward` 未合并时是 `result + F.linear(x, delta)`；`merge` 是 `weight.data += delta`（`fan_in_fan_out` 时先转置）。两者等价，因此本题让参考解分别返回 `delta` 和合并后的 `y`，判题同时检查"稀疏结构 + 数值前向"。

## 对拍口径

- `check_example`：种子 777、\(n=4\)、`scaling=25` 的固定坐标样例，同时钉死 `delta` 与 `y`。
- `check_index_selection`：三组形状/种子，用 `(delta != 0)` 反推坐标集合与 `randperm` 对拍，并逐点核对取值。
- `check_forward_is_merged_linear`：3D 批量、`b0` 有/无两条分支，断言前向等于"\(W_0+\Delta W\) 的普通线性层"。
- `check_zero_spectrum_identity`：peft 初始化频谱为 0，增量必须严格全零（`torch.equal`）。
- `check_scaling_is_linear`：`scaling` 翻倍时增量与前向增量同步翻倍。
- `check_seed_changes_support`：不同种子给出不同支撑集，确认种子真的被使用。

## 复杂度

`delta` 构造 \(O(d_\text{out}d_\text{in})\)（稠密矩阵本来就占这么多显存），但可训练参数只有 \(n_\text{frequency}\)；前向比合并后的普通线性层多一次稀疏增量项。
