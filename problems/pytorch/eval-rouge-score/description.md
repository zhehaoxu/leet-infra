# 评测：ROUGE-1 / ROUGE-2 / ROUGE-L

## 题目描述

ROUGE（Recall-Oriented Understudy for Gisting Evaluation，Lin, ACL 2004）是摘要与生成任务的主力指标。它不问"你说的对不对"，而问"**参考里有的内容你覆盖了多少**"——以召回为主、用 F1 平衡精度。ROUGE-N 数 n-gram 重合，ROUGE-L 数**最长公共子序列**（LCS）：后者天生对顺序敏感、却允许跳着匹配，这正是它与 ROUGE-2 的分工（连续 vs 顺序）。CNN/DailyMail、XSum 这类摘要数据集的官方分数就是它；RLHF 里的规则型奖励（rule-based reward）也常直接挂 ROUGE-L。

## 规则

- `hypotheses`：`list[list[int]]`，每条摘要是一串 token id（已切好，本题不用任何分词器）。
- `references`：`list[list[list[int]]]`，`references[i]` 是第 \(i\) 条假设的参考列表（至少 1 条）。
- 聚合口径：逐对（一条假设 × 它的一条参考）算 F1，**先在该假设的多条参考之间取最大 F1**，再**对所有假设取算术平均**。
- **ROUGE-N**（\(n \in \{1, 2\}\)）：设假设与参考的 n-gram 频次为 \(\text{count}_h\)、\(\text{count}_r\)，
  \[ O_n = \sum_{g}\min\big(\text{count}_h(g),\ \text{count}_r(g)\big),\quad
  R = \frac{O_n}{\sum_g \text{count}_r(g)} = \frac{O_n}{\max(0, |r|-n+1)},\quad
  P = \frac{O_n}{\max(0, |h|-n+1)} \]
  \[ F1 = \frac{2PR}{P+R} \]
  注意 \(O_n\) 是**裁剪重合**：假设里重复出现的 n-gram 不能白拿超过参考里次数的分。
- **ROUGE-L**：令 \(l = \text{LCS}(h, r)\)（必须自己写动态规划），
  \[ R = \frac{l}{|r|},\quad P = \frac{l}{|h|},\quad F1 = \frac{2PR}{P+R} \]
- 边界约定（都必须在实现里显式处理）：
  - 假设或参考为空列表 ⇒ 该对的 F1 记 \(0\)；
  - \(|h| < n\) ⇒ 该阶一个 n-gram 都没有 ⇒ F1 记 \(0\)（ROUGE-2 对单词摘要必然是 0）；
  - \(P = R = 0\) ⇒ F1 记 \(0\)，绝不出现 \(0/0\)；
  - 假设列表为空（候选集为空）⇒ 三个指标都是 \(0.0\)。
- 返回三元组 `(rouge1, rouge2, rouge_l)`，全部是 Python float。禁止使用 `rouge_score` / `nltk`（判题环境里也没有）。

## 示例

```
hypotheses = [[1,2,4,3], [1,2,5]]
references = [[[1,2,3,4]], [[1,2,3,4,5]]]

假设 0 = [1,2,4,3]  vs 参考 [1,2,3,4]
  ROUGE-1: O=4, R=4/4=1, P=4/4=1              -> F1 = 1
  ROUGE-2: h 的 bigram {(1,2),(2,4),(4,3)}，r 的 {(1,2),(2,3),(3,4)} -> O=1
           R=1/3, P=1/3                        -> F1 = 1/3
  ROUGE-L: LCS([1,2,4,3],[1,2,3,4]) = 3,  R=3/4, P=3/4 -> F1 = 0.75

假设 1 = [1,2,5]  vs 参考 [1,2,3,4,5]
  ROUGE-1: O=3, R=3/5=0.6, P=3/3=1            -> F1 = 0.75
  ROUGE-2: O=1（只有 (1,2) 命中）, R=1/4, P=1/2 -> F1 = 1/3
  ROUGE-L: LCS = 3（[1,2,5] 整条就是子序列）, R=3/5, P=1 -> F1 = 0.75

平均：rouge1 = (1 + 0.75)/2        = 0.875
      rouge2 = (1/3 + 1/3)/2       = 0.3333333333
      rouge_l = (0.75 + 0.75)/2    = 0.75
```

同一条假设 0：ROUGE-1 = 1 而 ROUGE-2 只有 \(1/3\)、ROUGE-L = 0.75——换序不伤 unigram，伤 bigram，LCS 居中。

## 约束

- CPU；token id 是任意非负整数，只比相等。
- 三个返回值都在 \([0, 1]\)。
- **退化性质**：假设与参考逐位相同 ⇒ 三项都是 1；完全不相交 ⇒ 三项都是 0；把某条参考复制一份 ⇒ 分数不变（取最大而非取平均）；单词摘要 ⇒ ROUGE-2 为 0 而 ROUGE-1/ROUGE-L 仍可有值。
- 与 BLEU 的关键差异：ROUGE 的 \(R\) 与 \(P\) **同时**出现在 F1 里，长输出伤 \(P\)、短输出伤 \(R\)，所以不需要额外的简洁惩罚。
