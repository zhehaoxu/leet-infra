import torch


def solve(postings, doc_lens, k=10, k1=1.2, b=0.75, block_size=4):
    """倒排 postings 的 skip-AND 求交 + Block-Max WAND top-k。

    入参：
    - postings: list/tuple，长度等于 query 的 term 出现次数；空 list 表示
      空 query。每个元素是 (n_i, 2) int64 张量，每行 (docid, tf)，要求
      docid 严格递增且 0 <= docid < N，tf >= 1；形状 (0, 2) 表示该 term
      不在语料。同一 term 出现多次就放多份相同的倒排表，分数按出现次数
      重复累加。
    - doc_lens: (N,) int64，N >= 1，0 <= doc_lens[d]；用于 BM25 的长度
      归一化。
    - k: int，默认 10，>= 0。
    - k1: float，默认 1.2，> 0；b: float，默认 0.75，落在 [0, 1]。
    - block_size: int，默认 4，>= 1。每条 postings 按连续 block_size 行
      切块（最后一块可不满），每块最后一条的 docid 就是 skip 目标。

    打分口径（BM25 + lucene idf，全部由入参现算）：
    - N = doc_lens.numel()，df_i = 第 i 条 postings 的行数，
      idf_i = log(1 + (N - df_i + 0.5) / (df_i + 0.5))；
    - avgdl = mean(doc_lens)；avgdl == 0 时长度比值约定取 0；
    - score(d) = 对所有「第 i 条 postings 含文档 d」的项累加
        idf_i * tf_{i,d} * (k1 + 1)
        / (tf_{i,d} + k1 * (1 - b + b * |d| / avgdl))。
      不在任何 postings 中出现的文档分数记为 0，但不进入 top-k。

    返回 (candidates, topk, pruned_blocks)：
    - candidates: (C,) int64，升序，同时出现在所有非空 postings 中的
      docid（AND 交集，用块尾 docid 作 skip 目标跳跃求交）。空 query 或
      任一条 postings 为空时 C = 0。
    - topk: (min(k, U),) int64，U = 至少出现在一条 postings 中的文档数；
      按 (score 降序, docid 升序) 排列。k = 0 时为空；k > U 时返回全部 U 个。
    - pruned_blocks: int，Block-Max WAND 中「整块跳过、块内没有任何文档
      被完整打分」的 postings 块数；被部分扫描过的块不计。空 query 或
      k == 0 时为 0。剪枝不得改变 topk 结果。

    非法输入一律抛 ValueError：postings 不是 list/tuple、元素不是 (n, 2)
    整型张量、docid 非严格递增或越界、tf < 1、doc_lens 形状或取值非法、
    N == 0、k < 0、k1 <= 0、b 不在 [0, 1]、block_size < 1。CPU、确定性、
    不调用随机数。
    """
    raise NotImplementedError
