# 分组查询注意力 GQA / MQA

## 题目描述

分组查询注意力（Grouped-Query Attention）是多头注意力（MHA）与多查询注意力（MQA）之间的折中：
**query 有 `num_heads` 个头，而 key/value 只有 `num_kv_heads` 个头**，每若干个 query 头共享同一组
K/V 头，从而大幅减小 KV cache 显存。Llama 2 70B / Mistral 等现代大模型都采用它。

## 规则

- 实现 `solve(q, k, v, num_heads, num_kv_heads)`。
- `q` 形状 `(B, Lq, num_heads*dh)`；`k`、`v` 形状 `(B, Lk, num_kv_heads*dh)`；返回 `(B, Lq, num_heads*dh)`。
- `num_heads` 必须能被 `num_kv_heads` 整除；第 `i` 个 query 头使用的 kv 头是 `i // (num_heads // num_kv_heads)`
  （**连续成组**，即 `repeat_interleave` 语义，而不是循环取模）。
- 缩放因子为 $1/\sqrt{dh}$，$dh = \text{末维}/num\_heads$；对每个 query 头独立做 $\text{softmax}(QK^\top/\sqrt{dh})V$ 后按头拼接。
- `num_kv_heads == num_heads` 时应与标准多头注意力一致；`num_kv_heads == 1` 时为 MQA。

## 示例

```
输入：q=[[[1,0,0,1],[0,1,1,0]]] (B=1,Lq=2,H=2,dh=2)
     k=[[[1,0],[0,1]]]        (Lk=2,KV=1)
     v=[[[2,0],[0,2]]]
     num_heads=2, num_kv_heads=1
输出：[[[1.339523, 0.660477, 0.660477, 1.339523],
        [0.660477, 1.339523, 1.339523, 0.660477]]]
```

## 约束

- float32 输入；$B \ge 1$，$Lq, Lk \ge 1$，$dh$ 为正整数
- 不使用因果掩码（本题只考分组共享 K/V 的机制）
