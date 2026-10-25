# 详解：logit 偏置与禁用 token

## 出处
- **llama.cpp（已回源，逐条对着源码原文核对）**：
  - `common/sampling.cpp` 的装配注释原文就是 `// logit bias: user biases + model suppress tokens (-INFINITY)`：先把 `params.logit_bias` 拷进一个 `merged` 向量，再对词表的 suppress tokens（`llama_vocab_get_suppress_tokens`）逐条 `merged.push_back({ suppress[i], -INFINITY })`，最后 `samplers.push_back(llama_sampler_init_logit_bias(...))`。**两步合成一个采样器、且排在 dry/top-k/top-p/min-p/xtc/temperature 那一堆 stage 之前**——这就是合集里 `llama-sampling-pipeline` 把 bias 记作 stage 0 的依据。
  - `src/llama-sampler.cpp::llama_sampler_logit_bias_apply` 原文：先走快路径 `if (lb.token >= 0 && cur_p->size > (size_t) lb.token && cur_p->data[lb.token].id == lb.token) cur_p->data[lb.token].logit += lb.bias;` 否则把这条塞进 `to_search`，再对候选数组做一次 `cur_p->data[i].id == lb.token` 的线性查找、命中才 `logit += lb.bias`。→ 题面「**加性**、按表顺序逐条、匹配不到就**什么也不做**」完全对应这段代码：越界/负 id 既不会报错也不会被裁剪。
  - 重复条目**不合并**：装配处只 `push_back`，`apply` 处对每条 `lb` 各加一次，所以 `[1,+0.5],[1,+1.5]` 就是 `+2.0`（题面易错点 3）。
  - `common/arg.cpp` 里 `-l, --logit-bias` 的用法串写作 `TOKEN_ID(+/-)BIAS`（示例 `--logit-bias 15043+1` / `15043-1`），`--ignore-eos` 的说明直接写着 `(implies --logit-bias EOS-inf)` → 「禁用某个 token」在 llama.cpp 里就是往偏置表里追加一条 `-INFINITY`，与题面 ①② 的关系正是「同一张表、一个特殊值」。
- **OpenAI 的 `logit_bias`**：Chat Completions 的 token→偏置映射，官方口径同样是**加在采样之前的 logits 上**，文档给出的取值区间是 \([-100, 100]\)、用 \(-100\) 表达"几乎不让它出现"。本次判题环境无法访问 `platform.openai.com`（403），逐字引文 **待回源**；但与本题的差别是**约定层面**的、可以断言的：OpenAI 只有加性偏置这一条通道，没有 \(-\infty\)，"ban" 只能靠一个足够大的有限负数——见核心易错点 1 的反例。
- **`--banned-rp` 这类独立开关不属于 llama.cpp**：在 `common/arg.cpp` 与 `common/sampling.cpp` 里 grep `banned` 无任何命中，硬禁用一律走 logit_bias 的 `-INFINITY`（suppress tokens / `--ignore-eos`）。立项材料提到的 `--banned-rp` 未能在 llama.cpp 源码中定位，**待回源**（可能来自别的引擎，故本题只把它当"硬掩码这一类做法"的通称，不作具体引用）。

## 完整推导与下标约定
记词表 \(V\)、原始 logits \(\ell\in\mathbb R^{V}\)（批量时逐行独立，最后一维是词表），偏置表为有序对序列 \(\big((t_1,\delta_1),\dots,(t_n,\delta_n)\big)\)（允许 \(t_i\) 重复、越界），禁用集合 \(B\subseteq\{0,\dots,V-1\}\)。

1. **偏置**。按 \(i=1..n\) 依次执行
   \[
   0\le t_i<V\ \Longrightarrow\ \ell_{t_i}\ \mathrel{+}=\ \delta_i,
   \qquad\text{否则跳过该条（不产生任何副作用）。}
   \]
   等价闭式：\(\ell'_t=\ell_t+\sum_{i:\,t_i=t}\delta_i\)，即同一 token 的偏置是**求和**；因此加法可交换、可合并，但**必须合并成求和**才等价，取最大值/后写覆盖都不对。
2. **掩码**。\(\ell''_t=\begin{cases}-\infty, & t\in B\\ \ell'_t, & \text{否则}\end{cases}\)。注意这是**赋值**，不是「\(\ell'_t+(-\infty)\)」之外的任何写法。
3. **softmax**。取 \(M=\max_t \ell''_t\)（要求每行 \(B\ne\) 全集，否则 \(M=-\infty\)、\(\ell''-M\) 全是 \(\infty-\infty=\text{NaN}\)，实测 `torch.softmax` 对整行 \(-\infty\) 返回 \([\text{nan},\text{nan},\text{nan}]\)）：
   \[
   p_t=\frac{\exp(\ell''_t-M)}{\sum_{t'}\exp(\ell''_{t'}-M)},\qquad
   \exp(-\infty-M)=0\ \Longrightarrow\ p_t=0\ (t\in B)\ \text{精确成立}。
   \]
   分母只对存活项求和，所以 \(p\) 就是「在存活词表上的 softmax」。两种写法在浮点误差内等价：先掩码再 softmax，与先做全词表 softmax、把禁用项清零、再除以剩余质量——实测最大绝对差 \(7.451\times10^{-9}\)（float32 舍入级）。
4. **与"softmax 之后动手"的关系**。logit 域加 \(\delta\) 恒等于概率域给这一项**乘** \(e^{\delta}\) 再归一化：实测 \(\delta=+0.5\) 时两条路径给到小数点后 6 位完全相同的 \([0.366791, 0.366791, 0.134935, 0.081842, 0.04964]\)（最大差 \(5.960\times10^{-8}\)）。而把同一个数字当成**概率上的加法**（\(\ell^{\text{wrong}}_1=p_1+0.5\) 再归一）会得到 \([0.28577, 0.506662, 0.105129, 0.063764, 0.038675]\)——这就是「bias 作用在 softmax 之前还是之后」的真正分界：**乘 \(e^\delta\)** 是对的口径，**加 \(\delta\)** 不是。
5. **采样**。\(\text{chosen}=\min\{t:\sum_{t'\le t}p_{t'}\ge u\}\)，沿**原下标顺序**；被禁项概率为 \(0\) 使 CDF 出现平台，首个 \(\ge u\) 必然跳过它们。\(u\) 为向量时逐行独立（需要把 \(u\) 广播成 \((\dots,1)\) 再比较）。

样例逐步：\(\ell=[1.0,0.5,0.0,-0.5,-1.0]\)、\(((1,0.5),(1,1.5),(3,-2.0))\)、\(B=\{4\}\)、\(u=0.6\) → \(\ell''=[1.0,2.5,0.0,-2.5,-\infty]\)，\(M=2.5\)，\(p=[0.170075,0.762222,0.062567,0.005136,0.0]\)，CDF \([0.170075,0.932297,0.994864,1.0,1.0]\) → 首个 \(\ge0.6\) 是下标 1。

## 核心与易错点
1. **"大负数 ≠ \(-\infty\)"，这是本题真正的坑**，两层都要说清：
   - *数值层*：同一份 \(\ell\)，把下标 4 用 \(-100\) 的偏置压下去，softmax 后它的概率是 \(2.802597\times10^{-45}\)（float32 次正规数，**不是 0**）；\(-30\) 是 \(5.76\times10^{-15}\)、\(-16\) 是 \(6.930466\times10^{-9}\)；只有 \(-\infty\) 给出**精确** \(0.0\)。有意思的是在 float32 下这三条的 CDF 末尾都停在 \(1.0\) 不动（平台），我对 \([0,0.999999999]\) 均匀扫 20 万个 \(u\) 也没能选中那个位置——所以「采样结果」看不出差别，差别只在返回的概率向量里（判题 `atol=1e-5` 同样看不出）。**别把这一点当成"−100 够用"的证据。**
   - *语义层*：加性偏置是**相对**量，"禁得住"取决于该 token 与其它 token 的**差距**，不取决于偏置的字面大小。实测反例：\(\ell=[1000.0,0.0,0.0]\)，对下标 0 施加 \(-100\) 这条"OpenAI 口径的 ban"之后 \(p=[1.0,0.0,0.0]\)——它照样以概率 \(1\) 被选中；换成 \(-\infty\) 硬掩码才得到 \(p=[0.0,0.5,0.5]\)。因此本题的 ban 必须是 \(-\infty\) 赋值。
   - 同一逻辑适用于 `masked_fill(torch.finfo.min)`：合集里 `causal-mask` / `llm-gqa-attention` 已经明写 \(-\infty\) 与 finfo.min 是两种口径，本题沿用 \(-\infty\) 这一支。
2. **偏置是加在 logit 上、掩码在偏置之后**。同一下标两条都命中时以 \(-\infty\) 为准（实测：下标 4 先 `+50` 再进 `banned` → `biased_logits[4] = -inf`）。由于 \(-\infty\) 是赋值、且 \(-\infty+\delta=-\infty\)，把两步调换在精确算术下结果相同——但题面口径是 ①→②，别依赖这个巧合（若掩码写成有限大负数，调换顺序就真的会改变结果）。
3. **重复条目求和、不合并**：`[1,+0.5],[1,+1.5]` ⇒ `+2.0`（实测两者 `biased_logits` 完全一致）。写成 dict 后"后者覆盖前者"会得到 \(+1.5\)，错。
4. **越界/负 id 是无副作用，不是异常也不是环绕**：偏置表 \([-1,7,999]\) 配 \([5.0,-5.0,100.0]\)、禁用表 \([-3,42]\)，在 \(V=3\) 上与空表逐位相同（实测）。写 `clamp`、取 `abs`、或让 `IndexError` 冒出来都是错的。
5. **softmax 的分母是存活集合**，不是全词表：实测把 \(\ell=[5,4,3,2]\) 的下标 0 禁掉，剩余是 \([0.0,0.665241,0.244728,0.090031]\)——后三项正是 \([4,3,2]\) 的 softmax，而不是"先算 \(0.665241\) 再不管它"。少除一次分母，整行概率和就不是 \(1\)。
6. **整行全禁不在口径内**：\(M=-\infty\) 时 `exp(-inf - (-inf)) = NaN`（实测），题面已要求每行至少留一个存活项；`torch.softmax` 对**单个** \(-\infty\) 项处理得很好（给精确 \(0\)），对**整行** \(-\infty\) 就给 NaN，所以别指望换 API 能绕过这条约束。
7. **不要就地震改入参 logits**：返回的 `biased_logits` 是新张量（参考解先 `clone`）。合集里凡返回"改造后的 logits"的题（`penalty-sampling`、`dry-sampling`）同样如此。
8. **批量与 \(u\) 的形状**：\((\dots,V)\) 时逐行独立，\(u\) 可以是标量、单元素或 \((\dots,)\)；逆变换要把 \(u\) 升到 \((\dots,1)\) 才能与 \((\dots,V)\) 的 CDF 比较，参考解用 `u.view(u.shape + (1,)*(dim - u.dim()))`，并用 `argmax` 取首个 \(\ge u\) 的位置、整行都不满足时回退 \(V-1\)。
9. **不要用 `torch.multinomial`**：合集口径一律手算 CDF 逆变换，随机数必须是显式入参。

## 与邻近题的区别
- **vs `penalty-sampling`**：最容易被混的一对，因为两者都"在 softmax 之前改 logits"。差别有三层：① **来源**——penalties 的三个系数作用在**由解码历史算出的** token 计数上（窗口 `penalty_last_n` 内的 n-gram 次数 `count`），每步都在变；本题的偏置表是**调用方静态点名**的 (token, bias) 列表，与历史无关。② **算子形状**——penalties 先做 `l<=0 ? l*repeat : l/repeat` 的**符号对称乘除**、再减 `count*freq + (count>0)*present`；本题只有**纯加法**（平移不改变其它 token 之间的相对关系，只改这一项与他项的差）。③ **产出**——`penalty-sampling` 明确不做采样、只返回惩罚后的 logits；本题返回三元并真采样一次。另外 llama.cpp 里这是**两个 stage**（bias 采样器先被 push 进链，penalties 在其后），顺序不是可交换的优化。
- **vs `dry-sampling`**：DRY 罚的是"与历史 n-gram 重复的续写"，惩罚沿匹配长度指数衰减，同样是历史驱动、且是乘性基准换算成加性的一项；与本题"点名加分"没有交集。
- **vs 截断/掩码类（`top-p-sampling` / `min-p-sampling` / `xtc-sampling` / `top-n-sigma-sampling` / `typical-sampling`）**：它们按**分布形状**决定谁出局（排序、累积质量、与最大概率之比、统计阈值），出局集合随每一步的分布漂移；本题按**调用方点名的 id** 出局，与该 token 当下概率多高毫无关系——这也正是易错点 1 里 \(-100\) 失效的原因。
- **vs 一般意义的 banned / bad-words / grammar 掩码**：形式上都写成 \(-\infty\)，共同点是"支持集受限"。本题的 `banned` 就是这类硬掩码，且刻意与 \(-100\) 型大负数区分开；差别在于 grammar/结构化约束的掩码**随已生成前缀变化**（需要状态与自动机），本题的掩码是一次性给定的常量列表。
- **vs `llama-sampling-pipeline`（同批在制）**：那题把 bias 放在管线 **stage 0**，后面还要串 penalties→dry→top-k→top-p→min-p→xtc→temp，此时"偏置在温度之前"意味着下游温度会把偏移再除以 \(T\)、有效概率因子是 \(e^{\delta/T}\) 而非 \(e^{\delta}\)，且存活集合要逐步记账。**本题只考 bias + ban 这一步的数学**：没有温度、没有截断，所以不存在这些顺序歧义，也不要求你装配管线。

## 对拍口径
参考解走张量路：`clone` → 逐条 `+=`（带 `0 <= tid < V` 判断）→ 命中 `banned` 写 `NEG_INF` → 手写 softmax（`torch.where(torch.isinf(shifted), 0, exp(shifted))`，保证被禁项**精确**为 \(0\)）→ `cumsum` + `argmax` 逆变换。判题 `_oracle` 完全另起一路：把 logits 展平成 Python 列表，逐条 `+=`，禁用项写 `None` 作 \(-\infty\) 的哨兵，用 `max(x for x in lg if x is not None)` 求平移量、`math.exp` + `math.fsum` 手工 softmax，逆变换用带 `acc >= uv` 的显式循环。除样例外还有：40 组随机全量对拍（偏置表里故意混入越界与负 id、随机 \(0\sim2\) 个禁用项）、重复条目累加与合并写法等价、被禁项恰为 \(0\) 且扫 \(u\) 永不选中禁令位、越界条目与空表逐位相同、空表时等于纯 softmax 且支持 \((3,6)\) 批量。统一 `allclose(atol=1e-5, rtol=1e-5)`，`biased_logits` 的 \(-\infty\) 用 `equal_nan` 语义比较。

## 复杂度
偏置与掩码各 \(O(n + |B|)\)（逐条哈希/线性查找在这里都算常数级），softmax \(O(V)\)，逆变换 \(O(V)\)（批量为 \(O(\text{rows}\cdot V)\)）。空间 \(O(V)\) 一份副本。llama.cpp 里 C++ 侧的"快路径 + 线性兜底"双查是为了在候选数组被打乱时仍能命中，本题输入已是完整词表向量，按下标直接寻址即可。

## 实践提醒
`logit_bias` 在生产里通常来自三类上游：用户的敏感词/黑名单、结构化输出的强制前缀（把 EOS 打进 \(-\infty\) 就是 `--ignore-eos`）、以及 agent 框架里"这一步别选这几个工具"。三者要的语义不一样：前两者要**真禁**（必须 \(-\infty\)），第三个往往只想**降权**（有限偏置）。把三者都写成 \(-100\) 是常见事故——要么禁不干净（见易错点 1 的 \(\ell=[1000,0,0]\) 反例），要么把"降权"做成了"消灭"。另外偏置表是**逐步重发**的：如果每步都带上同一条 \(\delta\)，效果会累积（历史里已生成的 token 被一压再压），这类 bug 在纯单步判题里看不出来，只有端到端才暴露。
