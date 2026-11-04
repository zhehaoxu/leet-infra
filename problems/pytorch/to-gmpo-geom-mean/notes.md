# 解题笔记

## 思路

官方 callsys/GMPO 的 `train_zero_math_gmpo.py` 三方账：

1. **compute_monte_carlo_advantages**（L407-421）：token 级奖励 → 轨迹级优势的三种归一——base/drgrpo 纯 dev、grpo 组内 std（unbiased）+1e-8；
2. **learning_step 六路 pg-loss 分支**（L673-831）：gmpo 家族的「符号翻转 + 保守截断 + token 几何平均」（gmpo / noclip 内定 1000 / seqclip 先 sum 再 clamp / without_norm 不除 mask.sum()）+ grpo 系的标准 PPO-clip（wider 的 0.67/1.49 硬编码、else 的 1±ε token 级 clip + masked aggregator + loss_masks 门控）；
3. **KL token 奖励组装**（L594-605）：-coef·(logps-ref)·mask + 行内最大 mask 下标 eos 注入（全 False 行 -1 负索引）。

上游不入判题域：vLLM actor 的 logps 采集、tokenizer/template（_apply_template）、critic 价值学习（ppo 分支的 value_pred/GAE）、strategy.backward/optimizer 调度、entropy 统计、kl3（K3 估计进 reg_loss 的 β 门控段）、reinforce_update 残笔（只赋 pg_loss_max 无 loss，跑不通的官方死代码）。

与现有题划界：
- **to-dapo-loss / to-gspo-loss / to-cispo-k3-loss**（428/429）：DAPO 的 token 级 clip 与动态采样、GSPO 的序列级 ratio、CISPO 的停梯度裁剪权重；本题是 GMPO 的**几何平均 ratio + 符号翻转保守截断**——clip 作用在「翻转后的 diff」上而非 ratio 上，且 loss 保持向量；
- **to-dpo-loss-backward**：偏好损失解析反向；本题是 on-policy 策略梯度正向账；
- **778（进位借位）等 cot 批**：CoT 推理系统账；本题回到训练目标侧。

## 关键实现点（官方口径逐条钉死）

- **MCA 的组切分**：`view(-1, num_samples)`——序列按组连续排列（每组 num_samples 条）；`repeat_interleave`（不是 `repeat`）把组统计量映射回每 token；grpo 除数是 `(std + 1e-8)`（1e-8 在分母内）；
- **sgn 翻转三步**：翻转 → clamp → `max(sgn·diff, clamp)` → 翻回。max 是**保守截断**：adv>0 时 sgn=-1，被 clamp 压小的负 diff 保留原值（不放大 loss）；等价说法：token 级对数比的几何平均被限制在「不超过 ±ε 的单侧松弛域」内；
- **几何平均的实现即账本本体**：`exp(diff_max[mask].sum() / mask.sum())`——先布尔索引取有效 token（展平）再 sum；除的是 `mb_response_masks.sum()`（有效 token 数）；without_norm 变体少这个除法、seqclip 变体把 sum 移到布尔索引处（标量化，clamp 作用在序列总 diff 上）；
- **noclip 的 cliprange=1000 是分支体首行赋值**——不读 self.args.cliprange，入参被忽略；
- **grpo else 的 loss 管线**：逐位 max → masked_aggregator(axis=1) → `(pg_loss * mb_loss_masks).mean()`。mean 版除 mask.sum()（token 归一）、sum 版除 constant_normalizer（官方 drgrpo 传 generate_max_length，长度归一）；loss_masks 是序列级 0/1 权重；
- **stats 三键**：amax/amin 是 `logprobs_diff.detach() * mb_response_masks` 的全局极值（**含 padding 位**——乘 0 后 padded diff=0 参与 min 竞争，DIFFS 有正 diff 时 min 恒 ≤ 0）；zero 计数 gmpo 系看 pg_losses 向量、grpo 系看 pg_loss_max；
- **KL 注入的负索引**：`masked_indices.max(dim=1).values` 对全 False 行给 -1，`rewards[arange, -1] += fin` 是 pytorch 的负下标写入——注入最后一个位置而非越界报错；`final_rewards.squeeze()` 把 [batch,1] 压成 [batch] 才能与行索引对齐；
- **batch>1 是官方原生 RuntimeError**：gmpo 系 `if mb_advantage >= 0` 布尔化多元素张量、grpo 系 `-adv[2] * ratio[2,5]` 广播失败——校验层不拦，传播。

## 测试设计（13 例）

advantage 3（R1 三 critic_type 基本账、R2 双组 num_samples=4 的组切分/错序、5 路校验 ValueError）、pg 7（gmpo 正负 advantage 的 sgn 路径 + stats、adv=0 的 >= 边界 [-0.0]、四变体差异含 noclip 入参旁证与 seqclip 的 amax==amin、grpo 家族四路含 wider 硬 clamp/sum 聚合/loss_masks=0 门控、BIG 长序列四值互异 + seqclip 标量 stats、batch=2 双族 RuntimeError、8 路校验 ValueError）、kl_reward 3（基本账、全 False 行 -1 负索引注入、3 路校验）。全部期望值由官方源码 exec 沙箱（find 边界抽取 MCA + 六分支体，`self.args.*` 参数化替换，torch 2.10.0 float64）实测钉死；24 组 fixture 含 batch=1 官方形状、BIG 序列、adv=0 边界、全 False mask 行。

## 回源偏差记录

- 官方 MCA 是方法且 `del response_masks`（入参未用）；本题去 response_masks、critic_type/num_samples 显式入参。
- 官方六分支内联 stats 三键 append（gmpo 系 `(pg_losses == 0)`、grpo 系 `(pg_loss_max == 0)`）；本题提 `_stats_record(stats, diff, mask, pg_loss, zero_target)` 助记函数，append 语义与逐分支目标逐条对拍。
- 官方 masked_aggregator 是类属性（drgrpo 时 lambda 包 masked_sum + constant_normalizer=generate_max_length）；本题 solve 层以 aggregator/constant_normalizer 参数构造等价闭包。
- 官方 stats/infos 横跨 mini-batch 循环累积；本题单次调用各 append 一条（沙箱同构）。
- `reinforce_update` 分支（L673-674）官方只赋 pg_loss_max 不设 loss、无 stats——不完整的死代码，不入题。
