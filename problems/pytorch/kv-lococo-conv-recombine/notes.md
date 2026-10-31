# 解题说明 · LoCoCo 卷积重算合并

## 一句话

H2O 负责「硬留」重击 token，剩下的一律不丢，而是沿 token 轴做一次卷积 + softmax，把它们软合并成 `mem_compress` 个摘要 token，再用 `normalizer` 与「代表 token 的 one-hot 选择」混合。

## 为什么沿 token 轴卷积

非重击 token 有 \(n=S-\textit{keep}\) 个，要压成 \(m<n\) 个摘要。直接把同位置的 K/V 取平均会丢掉「谁和谁相邻」，而 local attention 的经验是邻域内的 token 最相关。核长 21、`padding=10` 的 same 卷积让每个输出位置恰好覆盖整个局部窗口：

\[
c_{o,t}=\sum_{d,j}w_{o,d,j}\,x_{d,t+j-10}
\]

`x` 的前 \(D\) 个通道是 K、后 \(D\) 个是 V，所以一个输出通道可以同时读 key 和 value；`m` 个输出通道就是 `m` 个摘要 token。卷积后再沿 token 轴 softmax，每一行变成对全部非重击 token 的凸组合，`blend` 才敢说「合并」而不是「又选了一次」。

## 流水线（按形状）

1. `hh_ids`：前缀 \([0,S-\texttt{local\_len})\) 上取 `keep-local_len` 个最高分，并上末段 `local_len` 个，按 token 号升序；
2. `res_ids`：补集按原顺序取出，分数 top-`m`，再按 token 号升序——这个顺序就是 `residual_weight` 的行序；
3. 非重击 K/V 按 token 号升序拼成 `(H, 2D, n)`，same 卷积 → `(H, m, n)`，ReLU，token 轴 softmax → `conv_prob`；
4. `residual_weight` 是每行一个 1 的 one-hot，`blend = residual*(1-ν) + conv_prob*ν`；
5. `merged = [heavy_hitter; blend @ non_hh]`，两段拼起来恰好 `mem_size` 行。

## 容易写错的地方

- **截断不是取整**：\(m=\lfloor \texttt{mem\_size}(1-\texttt{rate})\rfloor\) 用的是 `int()`，`10*0.25=2.5` 得 2；`keep` 必须由 `mem_size-m` 反推，否则两段长度对不上。
- **mask 的顺序**：上游全程用 `masked_select`，结果按 token 号升序而非分数序。重击段如果按分数排序输出，`merged` 的前 `keep` 行就和上游不一致。
- **残差行序**：`make_residual_weight` 在 scatter 前先 `sort`，所以 one-hot 的第 \(r\) 行对应第 \(r\) 小的代表 token，而不是第 \(r\) 高的分数。
- **local 段恒留**：`topk` 只在前缀上跑，名额是 `keep-local_len`；末段直接 `mask[...] = True`。末段分数再高也不占前缀名额，这正是「最近窗口不能被驱逐」。
- **卷积后有 ReLU**：默认 `hidden_act='relu'`，`nn.Sequential` 里 conv 后面紧跟激活；漏掉它，负的卷积输出会让 softmax 权重变得不可解释。
- **bias 与训练**：上游 `Conv1d` 默认带 bias、`normalizer` 是 `nn.Parameter`。本题把 bias 视为 0、`normalizer` 当输入，卷积核也不训练——考的是合并口径，不是训练流程。

## 复杂度

每个头 \(O(m\cdot n\cdot K\cdot 2D)\) 的卷积加 \(O(m\cdot n)\) 的 softmax，再乘一次 \(O(m\cdot n\cdot D)\)；`m` 通常只有 `mem_size` 的一小半，远小于原来的 \(n\)。

## 出处

`VITA-Group/LoCoCo`（arXiv 2406.05317）的 `llama/memory_compressor.py`：

- L56-58 `mem_compress = int(...)`、`keep_hh_size = mem_size - mem_compress`、`local_len`；
- L69-70 `nn.Conv1d(in_size, hidden_size, kernel_size, padding=(kernel_size-1)//2)`；
- L76 `normalizer = Parameter(ones(1)*normalizer_init)`；
- L88-100 `make_residual_weight`：L94 topk、L95 sort、L96-98 scatter one-hot；
- L110-117 前缀 topk 只取 `keep_hh_size-local_len`，L116-117 末段置 True；
- L121-132 `masked_select` 收集重击/非重击（升序）；
- L147-152 `cat([k,v])` → rearrange → conv → softmax → `residual*(1-normalizer)+weight*normalizer`；
- L155-159 `matmul` 回 K/V 并与重击段 `cat`。

超参默认值见 `train.py` L87-98：`mem_size=128`、`local_len=0`、`hh_keep_rate=0.5`、`n_convlayer=1`、`kernel_size=21`、`hidden_act='relu'`、`normalizer_init=0.5`。

**本题自己的补充**：同分取小 token 号；固定 `n_convlayer=1`、`expand=1`、bias=0；非重击不足 `m` 时显式报错，而不是让 `topk` 抛底层异常。

## 相关题目

- `kv-h2o-heavy-hitters`：只做硬驱逐，没有重算；本题把被驱逐的部分软合并回来。
- `kv-wkvd-least-squares-merge`：用最小二乘求合并系数；本题让卷积从邻域里学权重。
- `kv-snapkv-pooling`：池化也是压缩，但窗口由观测注意力决定，且不做可学习混合。
