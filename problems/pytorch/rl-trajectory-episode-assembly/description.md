# RL 基建：多段 episode 轨迹拼装与截断 bootstrap

## 题目描述

rollout 侧吐给训练侧的是一条**多段混合**的记录流：一个 buffer 张量里首尾相接塞着若干条 episode（多轮对话每条轨迹是一段），critic 每步记下 \(V(s_t)\)，段末还要对终局观测额外跑一次价值头得 \(V(s_{end})\)。下游的优势/回报计算（见 `to-gae`）要的却是按 episode 对齐好的 \((a_t, r_t, V(s_t), done_t, next\_value_t)\)。中间这层**拼装**有两个经典 bug：

1. 把「步数上限 / 最长生成本」造成的**截断（truncated）**当成真终止、bootstrap 清零——价值回归系统性偏低，模型学会「只要拖到上限就是好结局」；按 Gymnasium 的 `TimeLimit.truncated` 约定，截断段的 next_value 必须是终局观测的 \(V(s_{end})\)，**只有真终止（terminated）才置 0**。
2. 天真地沿拼接数组 `roll(-1)` 取后继值——一条 episode 的末步会把**下一条 episode 首步的 value** 当后继，价值跨段泄漏。`done` 的作用正是在边界处禁止「取下条记录」，改走 bootstrap/0 分支。

本题只做拼装与对齐，**不推导 GAE 递推本身**（那是 `to-gae` 的职责）。

## 规则

输入七个一维量（CPU）：`episode_id`（int64 \((T,)\)，同一 episode 恰构成一段极大连续段，id 全局唯一、可乱序）、`terminated` / `truncated`（int64 \((T,)\) 0/1 flag）、`actions`（int64 \((T,)\)，透传）、`rewards` / `values`（float32 \((T,)\)，\(r_t\) 与 \(V(s_t)\)）、`final_value`（float32 \((K,)\)，\(K\) 为连续段数，按**出现段序号**索引的终局观测价值 \(V(s_{end})\)；真终止段的该项存在但不被使用）。输入保证：每段恰在末步带且只带一个 flag，其余步两 flag 均为 0。

拼装口径：

1. **训练序**：各段按 `episode_id` **升序**重排，段内保持原时间顺序。
2. **done**：段末步 \(=1\)、其余 \(=0\)；截断结束也算 done。
3. **next_value 三分规则**（对齐训练序）：段内非末步 → 同段下一步的 `value`；段末且 `truncated=1` → 该段出现段序号对应的 `final_value`；段末且 `terminated=1` → \(0\)。任何位置都不许把其他段记录的 value 当后继。

返回七元组（全部对齐训练序，`int64`/`float32` 见下）：`actions`、`rewards`、`values`、`dones`、`next_values`（各 \((T,)\)），`lengths`（\((K,)\)，训练序每段步数）、`starts`（\((K,)\)，训练序每段起点，exclusive 前缀和）。\(T=0\) 时返回七个 0 元素张量。禁止现成 RL 库。

## 示例

三段按出现顺序的 id 为 7、3、8（乱序）；第 0 段真终止、第 1、2 段被截断：

```
episode_id = [7, 7, 7, 3, 3, 8, 8, 8]
terminated = [0, 0, 1, 0, 0, 0, 0, 0]
truncated  = [0, 0, 0, 0, 1, 0, 0, 1]
actions    = [10, 11, 12, 20, 21, 30, 31, 32]
rewards    = [1.0, 0.0, 0.5, 0.5, 2.0, 1.0, 1.0, 0.0]
values     = [0.5, 0.625, 0.75, 0.25, 0.375, 0.125, 0.25, 0.3125]
final_value= [0.9375, 0.875, 0.625]        # 按出现段索引；段0 真终止，0.9375 不被使用
```

训练序按 id 升序排成 3、7、8 三段，输出：

```
actions     = [20, 21, 10, 11, 12, 30, 31, 32]
rewards     = [0.5, 2.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0]
values      = [0.25, 0.375, 0.5, 0.625, 0.75, 0.125, 0.25, 0.3125]
dones       = [0, 1, 0, 0, 1, 0, 0, 1]
next_values = [0.375, 0.875, 0.625, 0.75, 0.0, 0.25, 0.3125, 0.625]
lengths     = [2, 3, 3]
starts      = [0, 2, 5]
```

逐位置看 `next_values`：位置 1 是 id=3 段的截断末步，bootstrap 取 `final_value[1]=0.875`；位置 4 是 id=7 段的真终止末步，取 \(0\)；位置 7 是 id=8 段的截断末步，取 `final_value[2]=0.625`。若天真 `roll(-1)`：出现序里 id=7 末步会拿到 0.25（id=3 段首步的 value，跨段泄漏），id=3 末步会拿到 0.125（id=8 段首步）——都不是 0.875/0 的正确口径。

## 约束

- CPU；\(T \le 4096\)；\(K \ge 0\)；int64 判定用 `torch.equal`，float32 用 `allclose(atol=1e-5, rtol=1e-5)`。
- 不变量：\(\sum\) `lengths` \(=T\)；`starts[0]=0` 且相邻 `starts` 差恰为对应 `lengths`；`dones` 之和 \(=K\)；每个 done 位的 `next_value` 要么是该段 `final_value`（截断）要么恰为 \(0\)（真终止），绝不出现「邻段记录的 value」；非 done 位的 `next_value` 必等于训练序下一位（同段）的 `values`。
- 边界：单步 episode（首末同一步）合法；全截断、全真终止合法；\(T=0\) 走空约定；id 已按升序出现时训练序 = 出现序。
- 分工：本题只做**多段拼装**——分段、重排、done 与 bootstrap 的三分规则；GAE 反向递推 \(\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\) 及其退化参数探针在 `to-gae`（单段、末端隐式 \(V_T=0\)，即「全真终止」特例）；跨 rank 重分片与微批打包在 `rl-rollout-shard-repack`。拼装产出的每段 \((rewards, values, next\_values, dones)\) 恰是下游逐段调 GAE 的输入。
