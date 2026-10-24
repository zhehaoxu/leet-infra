# 解题思路

## 面试官意图

「梯度裁剪在数据并行/ZeRO 下怎么做才对」是区分「用过 clip_grad_norm_」和「懂它内部逻辑」的分水岭。
绝大多数人写单机版没问题，一到分片就错在**各卡用自己的局部范数裁**——本题正是逼你意识到裁剪的
scale 必须来自**全局范数**。

## 为什么不能各 rank 独立裁

裁剪的本质是：若整体梯度太长，就把它**等比**缩短到 `max_norm`，方向不变。分布式下参数分片在多个
rank，「整体梯度」= 所有分片拼起来的大向量。于是：

1. 先算全局范数 $\|\mathbf{g}\|_p = \big(\sum_r\sum_i |g|^p\big)^{1/p}$——这是一个**跨 rank 的 reduce**
   （每卡算自己分片的 $\sum|g|^p$，All-Reduce 求和，再开 $p$ 次方）。
2. 全局唯一的 $\text{scale}=\min(1,\ \text{max\_norm}/\|\mathbf g\|_p)$。
3. 每卡用**同一个 scale** 裁自己的分片。

若各卡用自己分片的局部范数算 scale，各卡缩放系数不同 → 拼回去的整体范数既不是 `max_norm` 也不是原值，
**方向被篡改**。`check_global_not_per_rank_independent` 用一个两卡例子（局部 0.6/0.8、全局恰好 1.0）
把这条固化：正确答案是两卡同乘 0.5，而不是各乘 0.5/0.6、0.5/0.8。

## 与 torch 实现的对应

`torch.nn.utils.clip_grad_norm_` 内部：把所有参数的 `.grad` 拼成 total_norm，再
`clip_coef = max_norm / (total_norm + 1e-6)`，`clip_coef_clamped = clamp(clip_coef, max=1.0)`，逐个乘回。
本题把 `+1e-6` 省略成 `min(1, max_norm/norm)` + 显式 zero 保护，语义等价且示例数值干净。分片时
`total_norm` 用第 4 步的跨卡求和替换单机范数即可（正是 `torch.distributed` 里做全局裁剪的标准姿势）。

## norm_type 与边界

- $p=2$ 是欧氏范数（默认）；$p=1$ 绝对值和；实践中也常用 $\infty$（本题不含）。
- `total_norm==0` → scale=1（否则会 0/0 出 NaN）。
- 已 `<= max_norm` → scale=1，不改。

## 判题口径

`test_cases.py` 用「先 `cat` 成大向量再算范数」的独立参考（与逐分片累加是不同推导路径）核对，
覆盖：全局 vs 独立裁剪的判别例、below-threshold 不裁、$p=1$、零梯度、**不等长分片**、以及返回结构
`(list, float)` 和不修改输入。
