# FP16 动态 Loss Scaling 单步

## 题目描述

FP16 能表示的最小正规数约 $6\times10^{-5}$，浅层梯度常小于它 → 反传时**下溢变 0**，训练停滞。解决办法：反向前把 loss 乘一个大整数 `loss_scale`（把梯度整体抬进 fp16 可表示区），更新前再除回去；一旦发现上溢（inf/NaN）就跳过这步、把 scale 调小；连续多步不上溢则把 scale 调大。这是 `torch.cuda.amp.GradScaler` 的核心逻辑。本题实现**一个优化步**的完整判定。

## 规则

实现 `solve(grads, loss_scale, growth_interval, consec_ok, backoff_factor=2.0, growth_factor=2.0)`：

- `grads`：fp32 梯度张量（已含 loss scaling 之前的真实梯度）。
- `loss_scale`：当前缩放因子（float，> 0）。
- `growth_interval`：连续多少步不上溢就增 scale。
- `consec_ok`：到本步为止**连续不上溢**的步数（scaler 的外部状态）。

单步流程（务必逐条实现）：

1. `scaled = (grads * loss_scale)` 再转 **fp16**。
2. `overflow =` 转出的 fp16 里**是否出现 inf 或 NaN**（`isinf | isnan` 任一为真）。fp16 上限 65504，超出即 inf。
3. 若 `overflow`：`skip_update=True`；`new_loss_scale = loss_scale / backoff_factor`；`new_consec_ok = 0`；`unscaled_grads = grads`（本步不更新，原样返回、不做反缩放）。
4. 否则 `skip_update=False`：
   - 令 `c = consec_ok + 1`。若 `c >= growth_interval`：`new_loss_scale = loss_scale * growth_factor`、`new_consec_ok = 0`；否则 `new_loss_scale = loss_scale`、`new_consec_ok = c`。
   - `unscaled_grads = (scaled_fp16.float() / loss_scale)` 转回 `grads.dtype`（先量化进 fp16 再除回，**会带入 fp16 舍入误差**——这正是本题要体现的精度损失）。

返回 **dict**：`{"overflow": bool, "skip_update": bool, "new_loss_scale": float, "new_consec_ok": int, "unscaled_grads": torch.Tensor}`。

## 示例

```
grads = [0.001, 0.002], loss_scale = 65536, growth_interval = 2000, consec_ok = 0
  scaled → fp16 不溢出
  overflow=False, skip_update=False, new_loss_scale=65536, new_consec_ok=1
  unscaled_grads ≈ [0.00100040435791, 0.00200080871582]   # fp16 往返的微小失真

grads = [1.0, 2.0, 100.0], loss_scale = 65536, ...
  1.0*65536=65536 > 65504 → inf，overflow=True
  skip_update=True, new_loss_scale=32768.0, new_consec_ok=0, unscaled_grads=[1.0,2.0,100.0]（原样）

grads = [0.001,0.002], loss_scale=65536, growth_interval=2, consec_ok=1
  不溢出且 c=consec_ok+1=2 ≥ 2 → 触发增长：new_loss_scale=131072.0, new_consec_ok=0
```

## 约束

- fp16 上溢判据只看 `scaled.to(float16)` 的 inf/NaN；不要用 `grads*loss_scale > 某阈值` 近似（阈值随 dtype 变，且 fp16 max 精确为 65504）
- `new_loss_scale` 为 python float；`new_consec_ok` 为 int；`unscaled_grads` dtype 同 `grads`
- 不修改输入 `grads`
