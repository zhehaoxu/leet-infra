# 分词：Unigram 后验路径采样

## 任务

实现 `solve`，输入 `text`、`vocab`、`log_scores`、`alpha`、`uniforms`，根据所有完整切分的全局后验分布采样一条路径，并返回归一化常数、该路径的对数概率和消耗的 uniform 数量。只需 Python 标准库。

这是一道具有**明确数学模型和随机数消费协议**的独立练习。它不是 Viterbi，也不是在每个位置对匹配词条做局部 softmax；不声称是已核验的 SentencePiece `Sample` 实现或其种子级复现。

## 输入、约束与错误

- `text` 为长度 0..12 的 Unicode 字符串，位置和长度按 Python Unicode **码点**计，而非 UTF-8 字节或字形簇。
- `vocab` 是长度 0..20 的字符串列表，词条 ID 就是其下标。合法词条长度 1..12，且非空、互不相同。**空词条或重复词条抛 `ValueError`，即使 `text` 为空也必须先验证词表。**
- `log_scores` 为与 `vocab` 等长的列表，每项是有限 Python `float`，范围 [-100, 0]；它们无需构成已归一化概率。
- `alpha` 是有限数，范围 [0, 4]。
- `uniforms` 是长度**恰为 `len(text)`** 的列表，每项为 Python `float`，满足 $0\le u<1$。不读取全局随机数生成器。
- 若非空 `text` 无法由词表完整覆盖，抛 `ValueError`。除明确要求报错的情况外，其余输入保证合法。
- 不修改任何输入；不规范化、不添加空格标记、不做未知词回退。词条按原字符串精确匹配，可以重复使用同一个 ID。

## 全路径分布

设 $\Pi$ 为所有完整切分路径的集合。路径 $\pi=(i_1,\ldots,i_k)$ 合法，当且仅当依次拼接对应词条恰好等于 `text`。定义：

$$
S(\pi)=\sum_{j=1}^{k}\mathrm{log\_scores}[i_j],\qquad
w(\pi)=\exp(\alpha S(\pi))
$$

$$
Z=\sum_{\pi\in\Pi}w(\pi),\qquad
\log P(\pi)=\alpha S(\pi)-\log Z
$$

当 $\alpha=0$ 时，每条**完整路径**权重相同，并不是每步的候选词条等概率。不得直接对可能下溢为零的整路径权重求和；应稳定计算对数归一化常数。

## 前缀递推

令 $n=\mathrm{len}(text)$，$L[t]$ 是前缀 `text[:t]` 的对数配分函数，$L[0]=0$。在已计算更短前缀后，定义：

$$
C_t=\{(i,s):s=t-\mathrm{len}(\mathrm{vocab}[i])\ge0,
\ text[s:t]=\mathrm{vocab}[i],\ L[s]>-\infty\}
$$

$$
L[t]=\operatorname{LSE}_{(i,s)\in C_t}
\bigl(L[s]+\alpha\,\mathrm{log\_scores}[i]\bigr)
$$

空候选集合对应 $L[t]=-\infty$。对非空有限数集合，稳定的 log-sum-exp 为：

$$
\operatorname{LSE}(x_1,\ldots,x_r)
=m+\log\sum_{j=1}^{r}\exp(x_j-m),\qquad m=\max_j x_j
$$

若 $L[n]=-\infty$，没有完整切分；否则 `log_partition` 为 $L[n]$。

## 精确采样协议

1. 从字符末端 $t=n$ 开始，`uniforms_used=0`。
2. 只保留 $C_t$ 中的候选：词条必须匹配当前位置的结尾，而且它之前的前缀可达。按 **vocab ID 升序**排列，不按长度或字典序。
3. 消费 `uniforms[uniforms_used]`，然后计数加一。即使只有一个候选，也必须消费一次。
4. 候选 $(i,s)$ 的条件概率为：

$$
p_t(i,s)=\exp\bigl(L[s]+\alpha\,\mathrm{log\_scores}[i]-L[t]\bigr)
$$

5. 按候选顺序累计概率，选择累计值**严格大于**当前 $u$ 的第一个候选；等号落在下一项。浮点累计在末尾有舍入误差时，允许最后候选兜底。不要额外给词条加权、排序或重新做局部 softmax。
6. 记录所选 ID，跳到它的起点 $s$，重复直到 $t=0$。最后逆序所选 ID，得到从左至右的 `ids`。

未消费的 `uniforms` 尾部忽略。随机数在**从右至左逐步抽样**时使用，不能把第一个 uniform 用于一次整路径 CDF 后忽略其他值。

每个可达候选的数学概率都严格为正，不能因指数下溢将其排除：`u=0` 必须选择第一个可达候选。可在对数域累计 CDF，并把 `log(0)` 视为负无穷。例如 `text="ab"`、`vocab=["b","a","ab"]`、`log_scores=[-100,-100,0]`、`alpha=4`、`uniforms=[0,0]` 应返回 `([1,0], 0.0, -800.0, 2)`。

## 返回值

返回四元 `tuple`：`(ids, log_partition, log_probability, uniforms_used)`。

- `ids`：Python `int` 列表，从左至右。
- `log_partition`：Python `float`，等于 $\log Z$。
- `log_probability`：Python `float`，等于所采路径的 $\alpha S(\pi)-\log Z$。
- `uniforms_used`：Python `int`，等于采样步数，也等于 `len(ids)`。

空 `text` 在验证词表后直接返回 `([], 0.0, 0.0, 0)`，代表唯一的空路径。浮点输出采用数值容差比较；路径、ID 顺序和消费数量必须遵守上述协议。

## 示例 1：均匀于完整路径

输入：

```python
text = "aaa"
vocab = ["a", "aa"]
log_scores = [-90.0, 0.0]
alpha = 0.0
uniforms = [0.6, 0.75, 0.0]
```

输出：

```python
([1, 0], 1.0986122886681096, -1.0986122886681096, 2)
```

共有 `a+a+a`、`a+aa`、`aa+a` 三条路径。末端 ID 0 的概率是 $2/3$，ID 1 是 $1/3$；`0.6` 选择末端 `a`。剩余 `aa` 的两个候选等概率，`0.75` 选择 `aa`，逆序后为 `[1, 0]`。最后一个 uniform 未使用。

## 示例 2：精确 CDF 边界

输入：

```python
text = "aa"
vocab = ["a", "aa"]
log_scores = [0.0, 0.0]
alpha = 0.0
uniforms = [0.5, 0.0]
```

输出：

```python
([1], 0.6931471805599453, -0.6931471805599453, 1)
```

第一个候选的累计概率恰为 `0.5`，不严格大于 `u`，因此选择第二个候选 `aa`。

## 示例 3：分数改变后验

输入：

```python
text = "aa"
vocab = ["a", "aa"]
log_scores = [-3.0, -0.1]
alpha = 1.0
uniforms = [0.5, 0.9]
```

输出：

```python
([1], -0.09726430062146395, -0.002735699378536058, 1)
```

两条完整路径的权重分别是 $\exp(-6)$ 和 $\exp(-0.1)$。对数配分函数可以为负数，不是路径概率。

## 示例 4：空文本

输入 `text=""`、`vocab=[]`、`log_scores=[]`、`alpha=1.0`、`uniforms=[]`，输出 `([], 0.0, 0.0, 0)`。若将词表改为 `[""]` 并相应提供一个分数，则应抛出 `ValueError`，不能先返回空结果。
