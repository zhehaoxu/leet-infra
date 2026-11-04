# OPD 题解笔记

## 官方账的关键细节（沙箱实测钉死）

1. **teacher 切片校验的双重性**：官方 `completion_logprobs = all_logprobs[
   len(prompt_ids):]` 后统一查 `len != len(completion_ids) or any None`——
   all_logprobs 过短时切片结果短、落进同一 ValueError，不 IndexError。
2. **右移账的不对称**：input 吃 completion[:-1]、target 吃完整 completion
   ——两者长度契约同为 len(prompt)−1+len(completion)；右移错一位的变异由
   target 的 [0]×4+completion 精确钉死。
3. **float32 是 PyTRIO schema 契约**（logprobs/advantages float32、target
   int64）——期望值带 float32 噪声（−0.60000002384…），测试 round(·,6)
   对齐。
4. **统计粒度**：reverse_kls 逐 token extend——mean/std 是全批 token 粒度
   （样例 6 token：mean −1/6、std √(1/42) 量级）；空序列跳过不产 datum
   也不产统计。
5. **空 datums 是 RuntimeError**：官方区分「没有可训练数据」（RuntimeError）
   与「数据形状非法」（ValueError）两类错。
6. **IDT 调度三处回绕**：schedule[step % len] 选任务、rows[(start+offset)
   % len(rows)] 取批、resolve 的 max-周期数乘回。7 步 2:1 周期 → m5/c2
   （第 7 步回绕 medical）。
7. **batch_for_task_step 用任务自己的 step 下标**：medical step 0 与 ceval
   step 0 都从各自 rows[0] 起步——与全局 step 索引取批的变异可区分。
8. **resolve 的 steps=0 语义**：保证两类数据都至少遍历一次——ceil(size/
   batch) 各自步数、除以各自 cycle 份额取 max 周期数、乘回周期长度。

## 与相邻题的划界

- `to-gkd-jsd-loss`：GKD 用 JSD 拟合教师前向分布（需要 teacher 前向 logits
  /分布）；OPD 只需 teacher 对具体序列的逐 token logprob（sequence 级打
  分）、KL 方向是 student−teacher 且直接当 advantage。
- `to-kl-controller`：RLHF 的 KL 惩罚是对参考策略的逐 token penalty 进
  pg loss；OPD 的"KL"是蒸馏信号本身、无参考策略、无 PPO clip。

## 沙箱搭建教训

- 抽 step 内核时**嵌套层级数错**：kernel 重排后 for seq 只加了 4 空格、
  拼在 for prompt 批循环后成同级 → for 体为空 IndentationError——kernel
  要按目标嵌套深度重新缩进（8 空格）。
- 官方函数的**类型注解也是依赖**：`args: argparse.Namespace` 在 exec 时
  求值，stub env 要给 argparse；`trio.ModelInput.from_ints` 出现在函数体
  内也要 stub。
- solution 初版把 seq 写成 dict 访问（seq["tokens"]）、fixture 是对象
  （seq.tokens）——官方表面是属性访问，跟着官方写。
- float32 噪声进 EXPECTED 前统一 round(·,6)，test 端同样 round 后比对。
