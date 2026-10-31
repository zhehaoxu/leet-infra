# 块扩散采样：一次定稿步里的两种配额口径

## 题目描述

块扩散语言模型（BD3-LM 一类的非因果并行解码）不是一次吐一个 token，而是把待生成的 \(L\) 个位置排进一个块，反复迭代 \(N\) 步：每一步网络对整个块都给出一份候选 token 和它的置信度，调度器决定**这一轮把哪些位置定稿**（commit），剩下的位置下一轮再看。定稿判据只有两个来源——置信度过阈值，以及把块平摊到各步的配额。

diffusers 的 `BlockRefinementScheduler.step` 在同一个函数里写了两套**互不相同**的配额口径，取决于是否传入掩码 token：

- **掩码填充**（给了 `mask_token_id`）：位置当前是不是掩码决定它能不能被定稿，本步的配额是「把 \(L\) 平摊到 \(N\) 步」这张表里**第 \(t\) 格**的定量值。
- **均匀破坏**（`mask_token_id=None`）：位置一开始就都填着随机 token，调度器自己记着哪些已经定稿（`committed`），配额是「前 \(t+1\) 步**累计**应该定稿多少个」，两者之差才是本步要补的。

这两种口径在多数 \((L,N)\) 下给出的逐步定稿数并不相同：\(L=10,N=6\) 时前者是每步 \([2,2,2,2,1,1]\)，后者的累计目标是 \([2,4,5,7,9,10]\)、即增量 \([2,2,1,2,2,1]\)——前者的前缀和是 \([2,4,6,8,9,10]\)，第 3、4 步各多要一个。把它们混用，块的收敛节奏就错了。

本题把这一步的离散决策单独端出来：网络采样已经做完（`sampled_tokens`、`sampled_probs` 直接给出），随机重加噪也做完（`random_tokens` 直接给出），你只负责算出新的块和两张定稿/编辑索引。没有 logits，没有 softmax，没有随机数。

## 规则

记 batch 为 \(B\)、块长为 \(L\)、`timestep` 为 \(t\)、步数为 \(N\)，\(p_i\) 是 `sampled_probs` 第 \(i\) 格、\(\tau\) 是 `threshold`、\(\tau_e\) 是 `editing_threshold`。所有与 \(p\) 有关的比较都**先把 \(p\) 降成 float32** 再做。

### 掩码填充模式（`mask_token_id` 不为 `None`）

每步配额表把余数前置：

\[ q_i=\Big\lfloor\frac{L}{N}\Big\rfloor+1\ (i<L\bmod N),\qquad q_i=\Big\lfloor\frac{L}{N}\Big\rfloor\ (i\ge L\bmod N),\qquad \text{quota}=q_{\min(t,\,N-1)} \]

\(A=\{i:\ \text{sample}_i=\texttt{mask\_token\_id}\}\) 是本行的待定位置，未待定位置的比较值取 \(-\infty\)（不是 0）。逐行独立地：

\[ S=\{i\in A:\ p_i>\tau\},\qquad
\text{transfer}= \begin{cases} S & |S|\ge \text{quota}\\[2pt] \text{前 }\min(\text{quota},|A|)\text{ 名的 }A\ (p\text{ 降序，并列按下标升序}) & |S|<\text{quota} \end{cases} \]

只有**整块一个掩码都没有**时（\(A=\varnothing\)）本步的 `transfer` 才恒为空；某一行没有掩码只会让那一行为空。

编辑只作用于**已经定稿、且不属于 prompt**的位置。先判开关：

\[ \text{editing 生效}\iff \tau_e\neq\texttt{None}\ \text{且}\ \tau_e>0 \]

生效时 \(\text{editable}=\neg A\wedge\neg\texttt{prompt\_mask}\)（`prompt_mask=None` 视作全 `False`），

\[ \text{editing}=\text{editable}\wedge(p_i>\tau_e)\wedge(\texttt{sampled\_tokens}_i\neq \text{sample}_i) \]

最后 `prev_sample` 以 `sample` 为底，在 `transfer | editing` 上换成 `sampled_tokens`，其余位置**原样保留**（掩码仍然是掩码）。

### 均匀破坏模式（`mask_token_id` 为 `None`）

\[ \text{base}= \begin{cases} \varnothing & t=0\\ \texttt{committed} & t>0 \end{cases},\qquad
\text{target}=\Big\lceil \frac{(t+1)L}{N} \Big\rceil=\frac{(t+1)L+N-1}{N}\ \text{(整除)},\qquad
\text{needed}=\max\big(0,\ \text{target}-|\text{base}|\big) \]

排名把 \(\text{base}\) 里的位置全当成 \(-\infty\) 排在最后，其余按 \(p\) 降序（并列按下标升序），名次从 0 起：

\[ \text{transfer}=\neg\text{base}\wedge\big(\text{rank}<\text{needed}\ \vee\ p_i>\tau\big) \]

\[ \text{editing}=\text{base}\wedge(\texttt{sampled\_tokens}\neq\text{sample})\wedge(p_i>\tau_e)
\quad\text{只要 }\tau_e\neq\texttt{None}\text{ 就生效（}0\text{ 与负数照样开）} \]

令 \(\text{after}=\text{base}\vee\text{transfer}\)：

\[ \text{prev\_sample}_i= \begin{cases} \texttt{sampled\_tokens}_i & \text{transfer}_i\ \text{或}\ \text{editing}_i\\ \text{sample}_i & \text{after}_i\ \text{但两条都没命中}\\ \texttt{random\_tokens}_i & \neg\text{after}_i \end{cases} \]

`committed` 与 `random_tokens` 在掩码填充模式下完全被忽略，`prompt_mask` 在均匀破坏模式下完全被忽略。

### 五个容易写反的地方

1. **两种配额不能互换**：掩码模式吃 \(q_{\min(t,N-1)}\) 这个**每步定量**，均匀模式吃 \(\lceil (t+1)L/N\rceil\) 这个**累计目标**。把后者写成前者的前缀和是错的——\(L=10,N=6\) 时前缀和是 \([2,4,6,8,9,10]\)，累计目标却是 \([2,4,5,7,9,10]\)。
2. **\(|S|\ge\) quota 时提交的是全部过阈位置**，不是「取前 quota 个」；\(q_i=0\)（\(N>L\) 的后段）也走这条分支，此时只要有过阈位置就全提交。
3. **\(t\) 越界不报错**：\(\min(t,N-1)\) 之后照用最后一格的配额；均匀模式 \(t\ge N\) 时 target 会超过 \(L\)，`needed` 大到把所有未定稿位置一次收完。
4. **\(\tau_e\) 的开关不对称**：掩码模式要求 \(\tau_e>0\)，\(0.0\) 与负数**关闭**编辑；均匀模式只判 `is not None`，\(0.0\) 与负数**开启**编辑。
5. **均匀模式最后那次 `random_tokens` 覆盖**发生在 `after` 上（本步刚定稿的位置已经受保护），且 \(t=0\) 传的 `committed` 必须当没用。

## 示例

### 例 1：掩码填充

```
L = 6, N = 4  ->  floor(6/4)=1, 6%4=2  ->  quota 表 = [2, 2, 1, 1]
t = 1  ->  quota = 2
mask_token_id = 9, threshold = 0.9, editing_threshold = 0.6
prompt_mask   = [False, False, True, False, False, False]

sample        = [[9, 9, 3, 9, 7, 9],
                 [9, 9, 9, 9, 9, 9]]
sampled_probs = [[0.93, 0.40, 0.99, 0.61, 0.88, 0.12],
                 [0.10, 0.93, 0.30, 0.95, 0.91, 0.05]]
sampled_tokens= [[4, 8, 1, 6, 2, 0],
                 [0, 5, 2, 7, 3, 8]]
```

第一行待定位置是 \(\{0,1,3,5\}\)，其中过阈的只有 \(0\)（0.93）——\(|S|=1<2\)，走 top-\(k\)：在 \(\{0,1,3,5\}\) 里按 \(p\) 取前 2 名，即 \(0\)（0.93）与 \(3\)（0.61）。第二行全待定，过阈的有 \(\{1,3,4\}\) 三个，\(|S|=3\ge 2\)，于是**三个全提交**。编辑只看非掩码位：第一行的 \(2\) 虽 \(0.99>0.6\) 且 token 变了，但被 `prompt_mask` 挡住；\(4\) 的 \(0.88>0.6\) 且 \(2\neq 7\) → 编辑。第二行没有非掩码位，编辑恒空。

```
prev_sample           = [[4, 9, 3, 6, 2, 9],
                         [9, 5, 9, 7, 3, 9]]
transfer_index        = [[True, False, False, True, False, False],
                         [False, True, False, True, True, False]]
editing_transfer_index= [[False, False, False, False, True, False],
                         [False, False, False, False, False, False]]
```

把 `t` 改成 5（\(>N-1\)）后 quota 仍取表尾的 1，第一行 \(|S|=1\ge1\) 于是只提交 \(\{0\}\)；把 `editing_threshold` 改成 `0.0`，本例的编辑立刻全空（开关要求 \(>0\)）。

### 例 2：均匀破坏

```
L = 6, N = 4, t = 2  ->  steps_done=3, target = (18+3)//4 = 5
threshold = 0.9, editing_threshold = 0.5, mask_token_id = None

committed     = [[True, True, False, False, True, False],
                 [False, False, False, False, False, False]]
sample        = [[7, 2, 9, 4, 5, 1],
                 [6, 3, 0, 8, 2, 4]]
sampled_probs = [[0.95, 0.20, 0.88, 0.61, 0.30, 0.72],
                 [0.10, 0.55, 0.70, 0.25, 0.92, 0.40]]
sampled_tokens= [[1, 9, 3, 6, 8, 2],
                 [5, 7, 4, 0, 9, 3]]
random_tokens = [[1, 8, 6, 2, 3, 9],
                 [7, 1, 5, 8, 0, 6]]
```

第一行已定稿 3 个 → needed = 2；未定稿的 \(\{2,3,5\}\) 按 \(p\) 排名是 \(2\)(0.88) 第 0、\(5\)(0.72) 第 1、\(3\)(0.61) 第 2 → 前两名进 `transfer`，\(3\) 既没过阈也没进配额。已定稿位上的编辑：\(0\) 的 \(0.95>0.5\) 且 \(1\neq7\) → 编辑；\(1\)（0.20）与 \(4\)（0.30）都不过 \(\tau_e\)。第二行 needed = 5，六个里最低名次的 \(0\)（0.10）落榜，其余五个全定稿。

```
prev_sample           = [[1, 2, 3, 2, 5, 2],
                         [7, 7, 4, 0, 9, 3]]
transfer_index        = [[False, False, True, False, False, True],
                         [False, True, True, True, True, True]]
editing_transfer_index= [[True, False, False, False, False, False],
                         [False, False, False, False, False, False]]
```

`prev_sample[0][3]=2` 来自 `random_tokens`：第一行第 3 格本步没定稿，被重加噪覆盖（原值是 4）。`prev_sample[0][1]=2` 与 `prev_sample[0][4]=5` 保持原样，因为它们在 `committed` 里却没命中编辑。

## 约束

- `sampled_tokens`、`sample`、`random_tokens` 必须是整数（非 bool、非浮点）张量且取值 \(\ge 0\)；`sampled_probs` 必须是浮点张量、有限、取值落在 \([0,1]\)；`committed`、`prompt_mask` 必须是 bool 张量。
- 前三者形状必须一致，为 \((B,L)\)，\(B\ge1\)、\(L\ge1\)；`committed`/`random_tokens` 非 `None` 时形状必须是 \((B,L)\)，`prompt_mask` 非 `None` 时形状必须是 \((L,)\)。
- 均匀破坏模式（`mask_token_id is None`）必须给 `committed` 与 `random_tokens`；掩码填充模式可以不给（给了也不参与计算）。
- `timestep`、`num_inference_steps`、`mask_token_id` 必须是 Python int（`bool` 非法，`mask_token_id` 可为 `None`）；\(t\ge0\)、\(N\ge1\)、\(\texttt{mask\_token\_id}\ge0\)。
- `threshold` 必须是有限实数；`editing_threshold` 是 `None` 或有限实数。
- 三个返回值 `(prev_sample, transfer_index, editing_transfer_index)` 的形状都是 \((B,L)\)，类型分别是 int64、bool、bool；逐位比对，无容差。违反任一入参约束抛 `ValueError`。禁止 `import diffusers`。
