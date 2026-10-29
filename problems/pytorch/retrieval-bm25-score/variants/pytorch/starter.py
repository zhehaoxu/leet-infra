import torch


def solve(docs, doc_lens, query, k=10, k1=1.2, b=0.75, idf_mode="lucene"):
    """BM25 全库打分与 top-k：df 与 avgdl 全部由语料现算。

    入参：
    - docs: (N, L) int64 整型张量，N 为文档数（N >= 1）。第 d 行前
      doc_lens[d] 个位置是文档 d 的 token id 序列（>= 0），其余位置是
      padding，内容任意（可为 -1），一律忽略；L 可以为 0。
    - doc_lens: (N,) int64，文档长度 |d|，满足 0 <= doc_lens[d] <= L。
    - query: (Q,) int64，Q >= 0，元素是 token id（>= 0）。空 query 合法。
      同一个 token 在 query 中出现多次时按出现次数重复累加，不做去重。
    - k: int，默认 10，>= 0；返回长度 min(k, N) 的 top-k。
    - k1: float，默认 1.2，> 0；词频饱和参数。
    - b: float，默认 0.75，落在 [0, 1]；长度归一化强度。
    - idf_mode: str，默认 "lucene"，只取 "lucene" 或 "robertson"：
      * "lucene":    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
      * "robertson": idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
      后者在 df(t) > N/2 时为负，必须按原式保留（不截断、不取
      max(0, idf)），因此 scores 中允许出现负值。

    统计与打分口径（不接受外部 df/avgdl，全部由入参现算）：
    - df(t) = 语料中包含 token t 的文档数，只看每行前 doc_lens[d] 个位置；
    - avgdl = mean(doc_lens)；avgdl == 0（所有文档为空）时长度比值
      |d| / avgdl 约定取 0，于是所有分数为 0，不产生 NaN；
    - 文档 d 对 query 中一次出现的 token t 贡献
        idf(t) * tf(t,d) * (k1 + 1)
        / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl)),
      其中 tf(t,d) 是 t 在文档 d 有效前缀内的出现次数；
    - score(d) = 对 query 的每一次出现累加上述贡献。query token 不在语料时
      df = 0、所有 tf = 0，贡献为 0（idf 仍按公式有定义）。

    返回 (scores, topk)：
    - scores: (N,) float64，全库每个文档的 BM25 分数（含 0 与负值）。
    - topk: (min(k, N),) int64，按 (分数降序, 文档下标升序) 排列的文档下标。
      k = 0 时为空张量；k > N 时返回全部 N 个。空 query 时所有分数为 0，
      topk 为 [0, 1, ..., min(k, N) - 1]。

    非法输入一律抛 ValueError：docs 不是二维整型张量、doc_lens 或 query
    形状/dtype 不符、doc_lens 越界、有效 token 或 query token 为负、
    N == 0、k < 0、k1 <= 0、b 不在 [0, 1]、idf_mode 未知。CPU、确定性、
    不调用随机数。
    """
    raise NotImplementedError
