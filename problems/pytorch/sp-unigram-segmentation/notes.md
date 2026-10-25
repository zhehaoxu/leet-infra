# 详解：Unigram 最优子词切分（Viterbi）

## 出处
Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*（ACL 2018，arXiv:1804.07389）。Unigram LM 给每个子词概率 \(p(x)\)，句子切分概率为子词概率连乘；编码取 \(\arg\max_S \prod p = \arg\max_S \sum \log p\)，用 Viterbi/最短路 DP 在 \(O(nL)\) 内求解（SentencePiece `Encode` 的确定解码即此 argmax；训练用 EM 估 \(p\)、用 unigram 概率做子词正则采样）。

## 核心与易错点
1. **全局最优 ≠ 贪心最长**：逐位贪心取最长词可能错过更优组合（如把一个高频长词拆成两个更高频词）；本题要的是 DP 全局最大。
2. **DP 递推**：`best[j]` = 前缀 `text[:j]` 的最大对数概率；`best[j]=max over 词表里能结尾于 j 的 p: best[j-len(p)]+logp(p)`（仅当 `text[j-len:j]==p` 且 `best[j-len]` 可达）。存 backpointer 记最优前驱词。
3. **可达性**：某前缀无法切分时置 \(-\infty\)；`best[n]==-\infty` 即无任何完整覆盖 → 抛 `ValueError`。
4. **唯一最优**：判题保证最优切分唯一，因此不必定义并列规则（实现里用严格 `>` 更新即可）；返回下标序列由回溯得到。

## 对拍口径
参考解为前向 DP（`best[]`+`back[]` 回溯成下标）。判题 `_oracle` **暴力枚举**全部 \(2^{n-1}\) 个间隙切法，逐个校验每片是否都在词表、算 \(\sum\log p\) 取最大——结构与 DP 完全不同，因最优唯一而结果一致；仅在极小串上跑（指数级）。另设「单片覆盖」「纯单字符回退」「不可覆盖抛 ValueError」「example 数值」断言。

## 复杂度
DP \(O(n L m)\)（\(n\) 串长、每位置试词表 \(m\) 个、比对长 \(\le L\)）；对拍 oracle \(O(2^n)\)。
