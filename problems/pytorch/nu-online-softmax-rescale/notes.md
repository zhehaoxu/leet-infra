# 详解：只交换两个标量为什么够，以及四类退化行为为什么要分开写

## 出处

「边扫边更新分母」这件事最早由 Milakov 与 Girouard 的《Online Normalizer Calculation for Softmax》(2018, NVIDIA) 讲清楚：softmax 的分母可以流式累加，代价只是每来一块就多一次缩放。FlashAttention (Dao 等, 2022) 把它变成 KV 分块里每步都要做的 rescale；另一条同样真实的线是词表并行——Megatron/DeepSpeed 的 `vocab_parallel_cross_entropy` 里每个 rank 只有 \(\text{vocab}/T\) 个 logits，交叉熵的分母必须靠一次 all-reduce 交换 (max, sumexp) 才能算对，本题的 `shard_size` 就是这个 rank 数的抽象。面试里能写出「段间只需要一个指数因子」并说清哪几类行必须提前分支，比背 FlashAttention 的伪码更能说明你真的动过分片的 logits。

## 一个指数因子为什么就够

设段内已经算好 \(l_s=\sum_j\exp(x_j-m_s)\)。整行分母想的是 \(\sum_j\exp(x_j-M)\)，两者只差一个常数：
$$\sum_j^{(s)}\exp(x_j-M)=\exp(m_s-M)\sum_j^{(s)}\exp(x_j-m_s)=\exp(m_s-M)\,l_s$$
因为 \(m_s\le M\)，缩放因子恒 \(\le 1\)，只会把数往小调、不会上溢。同理最后一遍算 `probs` 用 \(\exp(x_j-M)\)，参数也永远非正。整条链路上没有一次 `exp` 拿到正参数，这正是这类 kernel 数值安全的根源：不是「事后除一下」，而是每一步都相对某个真实存在的最大值。

顺带一个好用的性质：`probs` 对参考值的选择不敏感。若 \(M\) 取错了 \(M'\)，分子分母同时被 \(\exp(M'-M)\) 缩放，比值不变；而 \(\texttt{lse}=M'+\log\sum\exp(x_j-M')\) 在数学上也等于原值。所以上溢类 bug 往往在 `probs` 上看不见，只在 `lse` 上暴露——判题因此单独盯 `lse`。

## 四类退化行为要分开写

- **含 NaN 的行**：段内用张量 `max()` 取最大值，它会传播 NaN（`Tensor.amax` 的语义），于是 \(m_s\) 变 NaN、整行分流。
- **有 \(+\infty\) 的行**：\(M=+\infty\) 时所有有限项被 \(\exp(\text{finite}-\infty)=0\) 吸收，剩下的只是「几个 \(+\infty\) 均分 1」，所以先数个数再赋值，比让 IEEE 去算 \(\infty-\infty\) 稳。
- **整行 \(-\infty\)**：\(m_s=M=-\infty\)，于是 \(m_s-M\) 是 NaN。这种行必须提前分支直接输出「probs 全 NaN、lse \(=-\infty\)」，不能让 NaN 泼进整行的其余计算里——那正是「无定义」该有的表达方式。
- **某一段 \(-\infty\) 而行内有有限值**：\(\exp(-\infty-M)=0\) 且该段 \(l_s\) 本身也是 \(0\)，乘积干净，跳过即可。

## 核心与易错点

- **`torch.tensor(python float)` 默认 float32**：想拿它做段间归约，\(10^{39}\) 这种「有限但超出 float32」的最大值会被折成 \(+\infty\)，整行于是被送进 \(+\infty\) 支路，`lse` 从 \(10^{39}\) 变成 `inf`（实测 float64 输入下 `torch.logsumexp` 给 1e39，float32 输入给 inf）。段间归约请在 float64 上做。
- **Python `max(list)` 遇到 NaN 的结果取决于顺序**：`max([float('nan'), 1.0])` 与 `max([1.0, float('nan')])` 不一定同值，所以段间用 Python 归约时必须先把含 NaN 的行分流出去。
- **别用 `torch.exp` 的标量结果做 Python 浮点**：`torch.exp(torch.tensor(v))` 又踩回 float32；纯标量指数直接用 `math`。
- **不许先拼回整行**：判题虽看不住你内存里怎么做，但 \(10^{300}\) 级别的正参数 `exp` 会当场变 inf，测试里专门有一条断言「`probs` 不允许出现 inf」。
- **跨 `shard_size` 是末位差异**：切法改变累加顺序，float64 下差到 \(10^{-16}\) 相对量级，判题容差因此是 float64 1e-9 / float32 1e-5，而不是逐位。

## 与邻近题的区别

`softmax` 与 `log-softmax` 整行都在内存里，只需要减一次最大值；`logsumexp` 只求分母不求概率；`reduce-axis` 是朴素归约，没有跨块标定。`flash-attention-tile` 用的正是同一套 rescale，但它要再乘 V 并输出注意力，考点落在分块 GEMM 与掩码上。本题的独有切口是「按给定切法合并」以及 \(+\infty\) 行故意约定成均分而不是跟着 `torch.softmax` 给 NaN。

## 对拍口径与复杂度

oracle 完全不管切法：整行一次性 `torch.softmax(src, 1)` / `torch.logsumexp(src, 1)`，内部 float64，与参考解的「段内统计 + Python 层合并」是两条独立路径；NaN 用 `isnan` 断言，\(\pm\infty\) 用 `==` 短路，其余走绝对容差。另有独立用例覆盖「切法不变性」（shard 从 1 到 cols+2 全跑）、极端量级（\(\pm900\)、\(\pm1000\)）、特殊行四态、dtype/形状/空张量、非法入参。复杂度 \(O(rows\cdot cols)\)，跨段交换量只有每行 \(O(\lceil cols/\texttt{shard\_size}\rceil)\) 个标量。
