# 断点续训的连续性账

## 题目描述

「断点续训」的验收标准不是 loss 曲线看起来差不多，而是**逐位一致**：把 model、optimizer、scheduler、全局 RNG 四本账全部存档，从第 3 步恢复后继续跑 3 步，loss 序列必须和一口气跑 6 步完全相同。少恢复任何一本账（尤其是 RNG），后半程的数据流就整体错位。更要命的是 `Optimizer.state_dict()` 返回的张量与活体 momentum buffer 是同一对象，两次加载同一份 ckpt 会互相原地污染。本题跑三条路径，把恢复连续性、漏恢复 RNG 的发散、以及 ckpt 别名污染三件事全部做成可检验的账目。

## 规则

实现 `solve()`，按 starter docstring 跑三条路径并返回连续性账 dict：

1. **路径 1（黄金轨迹）**：`torch.manual_seed(1234)` 后新建模型/SGD(lr=0.1, momentum=0.9)/StepLR(step_size=2, gamma=0.5)，连续跑 6 步得 `losses_full`。
2. **路径 2（完整恢复）**：同样的流跑 3 步，存档 model/opt/sched/rng/step 五件套（优化器 state 必须逐张量深拷贝），再消耗 `torch.randn(2)` 的无关随机数，随后用全新对象加载 ckpt 并 `set_rng_state` 恢复 RNG，续跑第 3～6 步得 `losses_resumed`。
3. **路径 3（漏恢复 RNG）**：与路径 2 完全同构、同样消耗 2 个垃圾随机数、同样加载 ckpt，唯独不恢复 RNG，得 `losses_no_rng`。

单步循环顺序固定为 `zero_grad → randn 取数 → loss → backward → opt.step() → sched.step()`，第 \(i\) 步生效 \(lr=0.1\cdot 0.5^{\lfloor i/2\rfloor}\)。`losses_resumed` 与 `losses_full` 的比较用 Python 列表逐位相等；loss 数值对拍用 `atol=1e-5, rtol=1e-5`。

## 示例

```
输入：无
输出：
{
 "losses_full": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                 0.0066848150454461575, 0.05954299122095108, 0.5288273692131042],
 "losses_resumed": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                    0.0066848150454461575, 0.05954299122095108, 0.5288273692131042],
 "losses_no_rng": [0.5204203128814697, 0.6624248623847961, 0.0687742680311203,
                   1.0938212871551514, 0.1766050010919571, 1.190037727355957],
 "resumed_bitwise_identical": true,
 "no_rng_tail_diverges": true,
 "final_weight_identical": true,
 "momentum_restored": true,
 "sched_last_epoch_resumed": 6
}
```

注意前 3 项里 `losses_no_rng` 与 `losses_full` 逐位相同（存档前走同一条流），从第 4 项起彻底发散——2 个垃圾随机数让后续 \((x,y)\) 全部错位。

## 约束

- 三条路径各自从 `torch.manual_seed(1234)` 起流；loss 为 float32 `loss.item()`，对拍 `atol=1e-5, rtol=1e-5`；布尔与 `sched_last_epoch_resumed` 必须精确。
- `resumed_bitwise_identical` 是 Python 列表 `==`（逐位），不是 allclose。
- 存档/加载优化器 state 时若不做逐张量克隆，路径 3 会读到被路径 2 原地更新过的动量，`losses_no_rng` 与 `momentum_restored` 都会记错账。
