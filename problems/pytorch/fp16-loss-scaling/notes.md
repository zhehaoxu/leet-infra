# FP16 动态 Loss Scaling 详解

## 上溢与下溢是两回事

fp16 的**可表示区间窄**（正规数约 $6\times10^{-5}$ 到 65504）。混合精度里两处会出事：

- **loss 反传下溢**：小梯度直接变 0（最致命，浅层学不动）。loss scaling 就是解决这个——乘大数把梯度抬进可表示区。
- **scaled loss 上溢**：乘完 scale 若超 65504 → inf，污染后续。所以要检测 inf/NaN 并跳过该步。

本题只管**上溢检测 + scale 自适应**。「抬下溢」是它的收益，「可能上溢」是它的代价，scaler 就在两者间动态找最大可用 scale。

## 为什么溢出时不 unscale

一旦 `scaled` 出 inf，反量化回去（÷loss_scale）仍是 inf/NaN，用它更新权重会毁掉模型。所以标准做法：**本步作废**（skip optimizer step），只把 `loss_scale /= backoff`，下一轮用更小 scale 再试。因此本题溢出分支 `unscaled_grads = grads` 原样返回（不会被使用，仅为契约完整）。

## 增长的「连续」计数

`consec_ok` 是外部状态、每步 +1，遇溢出**清零**。只有连续累计到 `growth_interval`（默认 2000）才把 scale 翻倍并清零计数。逻辑放在「不溢出」分支里、且先 `+1` 再比较 `>=`，别把顺序写反（少一步或多一步会让 long-run scale 漂移）。

## fp16 往返误差是特性不是 bug

非溢出时 `unscaled = fp16(grads*scale).float()/scale` 与原始 `grads` 有微小差异（示例 `[0.001→0.001000404…]`）。因为值被「量化」进 fp16 的稀疏格点再回来。真实训练里这个误差被容忍——换来的是把小梯度从下溢 0 里救出来。判题**要求**复现这个 fp16 舍入（不是原样返回 grads）。

## 判据只信 dtype，别用阈值

fp16 max 精确是 65504（`torch.finfo(float16).max`）。`1.0*65536=65536>65504` 会 inf。用 `isinf|isnan` 判据而非自己设阈值，既准确又对 bf16/其它 dtype 可推广（bf16 范围同 fp32 几乎不上溢——这正是 bf16 免去 loss scaling 的原因，见延伸）。

## 关联

- 同族：量化「精度档位」主题——`fp8-quantize`（更极端的低位宽）、`per-channel/group-wise-quantize`（推理量化）。
- 延伸：bf16 有 fp32 的指数范围、通常不需 loss scaling；fp8 训练则需 per-tensor scaling（与量化族合流）。
- 判题：`check_example` 用 dump 的 fp16 往返值；另有溢出/backoff、增长触发、fp16 max 边界、不修改输入用例。
