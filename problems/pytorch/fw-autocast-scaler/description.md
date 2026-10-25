# autocast 与 GradScaler 训练循环账

## 题目描述

混合精度训练的稳定性挂在两件事上：autocast 决定每个算子跑在什么 dtype，GradScaler 决定溢出时「跳哪一步、scale 怎么升降」。`scale → backward → unscale_ → step → update` 的次序是状态机的一部分，错序会直接抛异常。本题在 CPU 上用 `torch.amp.autocast('cpu', dtype=torch.float16)` + `torch.amp.GradScaler` 跑 6 步（第 1、4 步注入 inf 梯度模拟溢出），要求把 dtype 账、scale 升降轨迹、跳步计数、权重轨迹全部记下来。

## 规则

实现 `solve()`，按 starter docstring 的固定构造与 8 步循环顺序执行并返回账目 dict。要点：

1. 所有数值都是二进制可精确表示的小数，fp16 前向没有舍入误差——loss/权重轨迹可用纯 float 逐步重算。
2. scale 状态机：发现 inf → 跳过该步更新、\(scale\times0.5\)、growth tracker 清零；连续成功步让 tracker 走到 `growth_interval=2` 时 \(scale\times2\) 并清零。
3. `unscale_` 后同一周期再调一次必须 `RuntimeError`（题面第 4 步的次序违规探针），捕获后流程照常。
4. 「权重是否被更新」用 step 前后 `torch.equal` 判定——CPU 上 `scaler.step()` 的返回值不能作为跳步信号。
5. 记下 `out/loss/grad` 三处 dtype 字符串：autocast 下 matmul 输出 fp16，`w.grad` 却是 fp32（autocast 的 cast 边界把梯度还原回参数 dtype，scaling 乘法也保持 fp32）。

## 示例

```
输入：无
输出：
{
 "out_dtype": "torch.float16", "loss_dtype": "torch.float16",
 "grad_dtype": "torch.float32",
 "scale_traj": [16.0, 8.0, 8.0, 16.0, 8.0, 8.0],
 "trackers": [1, 0, 1, 0, 0, 1],
 "updated": [true, false, true, true, false, true],
 "losses": [0.140625, 0.0087890625, 0.0087890625, 0.00054931640625,
            3.4332275390625e-05, 3.4332275390625e-05],
 "n_skipped": 2,
 "final_w_flat": [0.72412109375, 0.17529296875],
 "double_unscale_raises": true
}
```

推演 t3：t1 跳步使权重停在 t0 后的值，t2 起用 scale 8；t2、t3 两个连续成功步让 tracker 从 1 走到 2，触发翻倍回 16 并清零。

## 约束

- `scale_traj/trackers/updated/布尔/dtype 字符串`精确对拍；数值 `atol=1e-5, rtol=1e-5`。
- 固定超参与注入步，无随机数。
