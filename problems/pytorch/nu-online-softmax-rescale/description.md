# 分块 softmax 的重标定合并：只交换 (max, sum) 也能算对

## 题目描述

一行 logits 太长、被切成若干连续段时（FlashAttention 的 KV 分块、词表并行下每个 rank 只持有自己那一片 logits），没有任何一个 rank 看得见整行，但 softmax 的分母是整行的事。解决办法是让每段只交出两个标量——段内最大值与段内相对指数和——合并方用一个指数因子把它们标定到同一个参考最大值上。本题要求你按给定切法实现这套「段内统计 + 段间重标定」，同时输出概率与 log-sum-exp，并且全程不允许出现对上溢量取 `exp` 的写法。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 不许调用 `torch.softmax` / `torch.logsumexp` / `log_softmax`，也不许把整行拼回去一次算完——那样就没有重标定过程。
- 每行按 `shard_size` 切成连续段，最后一段可以短；`shard_size >= cols` 时整行就是一段。对第 \(r\) 行的第 \(s\) 段：

```
m_s = max(seg)                    segment maximum
l_s = sum(exp(seg - m_s))         relative to m_s, so one term is exp(0) = 1
M   = max_s m_s
L   = sum_s l_s * exp(m_s - M)    the one rescaling factor per segment
probs[j] = exp(x[j] - M) / L
lse = M + log(L)
```

- 取 `exp` 的参数必须永远 \(\le 0\)：段内先减 \(m_s\)，段间只乘 \(\exp(m_s-M)\le 1\)，最后只算 \(\exp(x_j-M)\le 1\)。出现 \(\exp(900)\) 这种写法就是错的。
- 逐行的特殊约定：含 NaN → 该行 `probs` 全 NaN、`lse` 为 NaN；有 \(k\ge1\) 个 \(+\infty\) → 这 \(k\) 个位置各得 \(1/k\)、其余 \(0.0\)、`lse` \(=+\infty\)（`torch.softmax` 在这种行上给 NaN，本题故意约定成均分，工程上更需要「不炸」）；整行 \(-\infty\) → `probs` 全 NaN、`lse` \(=-\infty\)；某一段全 \(-\infty\) 而行内另有有限值 → 该段贡献 \(0\)；最大值有限但超出 float32 范围（如 \(10^{39}\)）→ 仍是普通行，`lse` 必须保持有限。
- 校验：\(x\) 必须是二维 float32/float64 张量且 `cols >= 1`，`shard_size` 必须是 \(\ge1\) 的整数（`True`、`1.5`、`'2'`、`None` 都算非法），否则抛 ValueError。`rows == 0` 返回形状正确的空结果。

## 示例

记 \(\ln 2=0.6931471805599453\)、\(\log 3=1.0986122886681098\)，`shard_size = 2`：

```
x = [[0.0, 0.0, 0.0],
     [0.0, ln2, -inf]]

probs = [[1/3, 1/3, 1/3],
         [1/3, 2/3, 0.0]]
lse   = [log3, log3]
```

第一行两段分别贡献 \(l_1=2\)、\(l_2=1\)，\(M=0\)，所以 \(L=3\)。第二行 \(M=\ln2\)，第一段的 \(l_1=e^{-\ln2}+e^{0}=1.5\)，第二段全 \(-\infty\) 被跳过，于是概率是 \([0.5/1.5,\,1/1.5,\,0]=[1/3,2/3,0]\)，\(\texttt{lse}=\ln2+\log1.5=\log3\)。换成 `shard_size` 为 1 或 5，结果只在最后一位上不同。

## 约束

- 内部一律用 float64 计算，输出转回输入的 dtype；`probs` 与 \(x\) 同形状，`lse` 形状 (rows,)
- oracle 与分块算法完全无关：整行一次性调 `torch.softmax` / `torch.logsumexp`（float64）
- 跨 `shard_size` 只要求末位一致：float64 输入容差 1e-9，float32 输入容差 1e-5，因此不要指望逐位复现
- 每行段数为 \(\lceil cols/\texttt{shard\_size}\rceil\)，时间 \(O(rows\cdot cols)\)
