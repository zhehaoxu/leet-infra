# 累积和扫描详解

## 从归约到 scan：保留中间结果

`sum` 只给「全轴的总和」这一个数，而 `cumsum` 把每一步部分和都留下：
$$ \text{out}_i = \sum_{j\le i} x_j,\qquad \text{out}_i = \text{out}_{i-1} + x_i $$
它是**结合律算子** $(\mathcal S, +)$ 上的前缀扫描。把 $+$ 换成 `max/prod/min` 就是各自的
scan（如 `cummax` 用于数值稳定 softmax 的 running max）。

## 并行视角：scan 不是只能串行

朴素递推是 $O(n)$ 串行的，但前缀和满足结合律，可用 **Blelloch / work-stealing** 对数深度
并行化——这正是现代 GPU/`torch.cumsum` 与 Mamba 的「选择性状态空间 scan」能高效并行的
根本原因。串行实现便于理解语义，工程实现追求并行。

## 典型用途

- **top-p / 核采样**：对排序后的概率做 `cumsum`，找首个超过 `p` 的位置作为截断点
  （与本合集「top-p 采样」一题直接呼应）。
- **因果/累积掩码**：注意力里用位置索引的前缀比较构造下三角掩码。
- **积分/归一化**：把直方图转成分布函数（CDF），Gumbel / 逆变换采样靠 `cumsum` 得 CDF。

## 边界提醒

- `axis` 可为负，记得归一化成正轴再切片。
- 累积和的**末元素 = 整轴 sum**（见 `check_last_is_total`），可作自检不变量。
- 数值上长轴累加有漂移，浮点容差用 `allclose` 判断。
