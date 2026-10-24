# 解题说明 · 重复惩罚 (Penalties)

## 三种惩罚如何叠加

llama.cpp 把历史上常配的 `repeat_last_n / frequency_penalty / presence_penalty` 合并进
一个 `penalties` 采样器（源码 `llama_sampler_penalties`）。对窗口内出现过的每个 token：

```cpp
if (logit <= 0) logit *= penalty_repeat;   // 负 logit 放大 → 更负
else            logit /= penalty_repeat;   // 正 logit 缩小 → 更小
logit -= count * penalty_freq + (count > 0) * penalty_present;
```

- **乘法项** `penalty_repeat` 的对称设计很关键：无论 logit 正负，操作都让 logit **变小**
  （正数除以 `>1` 的数变小；负数乘以 `>1` 的数更负）。这样 softmax 后概率必然下降。
- **freq 项**随出现次数线性增长，专治「高频复读」。
- **present 项**是 0/1 阶跃，只要露过面就压一点，专治「刚说过就不想再说」。

## 窗口 = penalty_last_n

`token_count` 只统计最近 `penalty_last_n` 个 token（源码用增删维护滑动窗口：新 token
`++`、掉出窗口的 `--`，归零则 erase）。传 `-1` 表示不限制窗口（整段 history）。所以
`win = history[-penalty_last_n:]`（当 `penalty_last_n<0` 用整段），比 Python 负索引更
稳妥的写法是显式判断。

## 实现要点

- **顺序敏感**：必须**先**做 repeat 乘/除、**再**减 freq/present。反过来会让 freq/present
  也被 repeat 缩放，结果不同。
- `count > 0` 恒真（只有出现的 token 才会进计数表），present 项对每个被惩罚 token 恰好减一次。
- 只返回惩罚后的 logits，**不做 softmax / 采样**——这正是它在管线里所处的位置（softmax 上游）。
- 建议对 `logits.clone()` 就地改，避免动到入参。
