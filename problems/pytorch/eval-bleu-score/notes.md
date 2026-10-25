# 详解：语料级 BLEU

## 出处
Papineni, Roukos, Ward, Zhu, *BLEU: a Method for Automatic Evaluation of Machine Translation*（ACL 2002）。原文的三个式子：
1. 精度 \(p_n = \dfrac{\sum_{g\in n\text{-grams}}\text{Count}_{\text{clip}}(g)}{\sum_{g'\in n\text{-grams}}\text{Count}(g')}\)，\(\text{Count}_{\text{clip}}(g)=\min\big(\text{Count}(g),\ \max_{i}\text{Count}_i(g)\big)\)，\(i\) 遍历**同一句**的参考（原文式 (2)）。
2. 简洁惩罚 \(\text{BP}=\min\big(1,\ e^{1-r/c}\big)\)，\(c\) 为译文总长、\(r\) 为"有效参考长度"之和（式 (3)(4)）。
3. 组合 \(\text{BLEU}=\text{BP}\cdot\exp\big(\sum_{n=1}^{4}\lambda_n\log p_n\big)\)，原文对 \(\lambda\) 只做理论（order-N Markov 熵下界）论证，工程默认即本题的均匀权重 \(\lambda_n = 1/4\)。

工业实现：`nltk.translate.bleu_score.corpus_bleu`（默认 13a 分词、uniform weights、`method1` 平滑可选）、`sacrebleu`（自己管分词、默认 `exp` 平滑）、`torchtext` 的 BLEU、`tensor2tensor`/`tf.contrib` 的 `BLEUScore`。它们的分数彼此能差 1~2 个点，原因就是下面这几个约定不统一；本题把这些约定全部**写死在题面里**，并用 token id 输入绕开分词差异。

## 核心与易错点
1. **裁剪是按"每条假设 × 每个 n-gram"做的**：分子对 \(g\) 取 \(\min(\text{count}_h(g), \max_j \text{count}_{r_j}(g))\)。若不裁剪，输出 `the the the the` 对着参考 `the cat sat mat` 能拿到 \(p_1 = 1\)——这正是原文举的反例。若把 \(\max_j\) 错写成跨语料的 max，则会借到别的句子的参考频次，分数虚高。
3. **分母 \(C_n\) 数的是 n-gram 的出现次数 \(L-n+1\)，不是不同 n-gram 的种类数**；\(L<n\) 时该句贡献 0，别写成 1 或负数（\(\max(0, L-n+1)\) 而不是 \(L-n+1\)）。
3. **分母 \(C_n\) 用的是 n-gram **个数**（\(L-n+1\)），不是不同 n-gram 的种类数**；\(L < n\) 时该句贡献 0，别写成 1 或负数。
4. **有效参考长度（closest，tie → shorter）**：\(r\) 不是最短参考、也不是平均参考长度。原文用"与译文长度最接近的参考"，因为多条参考本身长短不齐，最短会过度惩罚、平均会把没必要的长度算进去。并列取更短是本题的确定性约定（NLTK 里靠遍历顺序实现，行为与"取更短"一致但依赖输入顺序）。
5. **BP 只在 \(c<r\) 时生效**：\(c=r\) 处指数恰好为 \(e^0=1\)，所以 \(c\ge r\) 分支写不写 min 都一样；但 \(c>r\) 绝不能出现 BP > 1，否则"输出堆长度"能刷分。
6. **零精度与平滑**：几何平均里只要有一个 \(p_n = 0\)，总分就归零——短测试集上 4-gram 常常一次都不命中，于是整句 BLEU = 0，这就是所有平滑方法存在的原因。本题选**最直白**的 Laplace 加一 \((M+1)/(C+1)\)，而不是 Chen & Koehnen (1999) 的 `method1`（\(1 + \sum_{k=n+1}^{N} p_k\) 与 \(\max(1,p_n)^{\gamma/(n-1)}\) 相乘）也不是 sacrebleu 的 `exp` 平滑；换实现时对不上是正常的，**判题只认题面口径**。另外要注意：\(C_n = 0\)（该阶在整个语料里都没有 n-gram）与 \(M_n = 0\)（有 n-gram 但一个都没命中）是两回事——前者加一后仍取 \(0\)，后者才被平滑救活。NLTK 对前者是用 \(\epsilon=10^{-9}\) 兜住 \(\log 0\)，口径不同但结论一致：这种语料不该给高分。
7. **别用字符串分词**：判题环境没有 `nltk`/`sacrebleu`，而且 13a 规则、标点、大小写都会改变 n-gram 集合。本题输入是 id 序列，把"分词口径"这个最大的不确定性从题目里移除了——实际项目里换 tokenizer 一定要重跑基线。

## 与邻近指标的区别
- **BLEU vs ROUGE**（`eval-rouge-score`）：BLEU 只有精度 + 长度惩罚，天然偏好"少而准"；ROUGE 以召回为主、用 F1 平衡，偏好"覆盖参考内容"。同一条过度简洁的译文：BLEU 可能很高，ROUGE-1 召回会掉。
- **BLEU vs token accuracy**（`eval-token-accuracy`）：都需要外部参考，但后者是逐位一对一比较、没有 n-gram 与裁剪概念；两者的共同点是都必须**语料级累加**（micro）。
- **BLEU vs PPL**（`eval-perplexity-logits`）：PPL 是 intrinsic 概率指标、不需要参考、可跨数据集比较趋势；BLEU 是 extrinsic 近似、强依赖参考集与训练期的评测口径。

## 对拍口径
参考解用 `collections.Counter` 一次性算出每条假设/参考的 n-gram 频次表，再对每个不同 gram 取 \(\min(\text{hyp}, \max\_\text{ref})\) 累加；几何平均走 `exp(mean(log p))`；BP 走分支。判题 `_oracle` 是完全另一套：显式三重循环手工切片拼 tuple、普通 dict 累加频次、内层对每条参考重新数一遍出现次数求 max，几何平均改用 `math.prod(prec) ** (1/max_n)`，BP 用 `min(1, exp(...))`（结构不同、数值等价）。除样例外另有：完美译文 → 1.0、重复 n-gram 的裁剪（`[7,7,7,7]` vs `[7,7]` → \(p_1=0.5\)）、\(p_4\) 为 0 时未平滑 BLEU 为 0 而平滑后 > 0、空语料与空译文、BP 饱和（\(c>r\)）、参考复制不变性，以及若干随机语料在 \(max\_n \in \{1,2,3,4\}\) × 平滑开关下的全量对拍，统一 `allclose(atol=1e-5, rtol=1e-5)`。

## 复杂度
设语料假设总长 \(T_h\)、参考总长 \(T_r\)：逐阶遍历为 \(O(max\_n \cdot (T_h + T_r))\) 的 n-gram 生成与哈希（tuple 构造本身 \(O(n)\)），排序/比较都没有；空间 \(O(\max_n)\) 个 n-gram 频次表。

## 实践提醒
BLEU 在小测试集（< 500 句）上方差极大，报数请带 bootstrap 置信区间；多参考时 \(r\) 的选择、以及"每条假设的参考数不同"都会引入不可比性——同一篇论文里换评测脚本就等于换了指标。