"""Chain of Agents 的串行接力账。

把 CoA（arXiv 2406.02818 Algorithm 1）的长文本处理摊成一份可逐项核对的
账本：分块 → 逐块串行接力（每块只看到**上一个** CU）→ manager 只用最后
一个 CU 出答案。

你实现的 `solve` 返回：

```
{
  "num_chunks": int,            # 块数
  "chunks": [{"idx", "text", "words", "start", "end"}],
  "order": [int],               # worker 处理顺序（块下标）
  "cus": [str],                 # 每轮接力后的 CU，按处理顺序
  "workers": [{"chunk_idx", "chunk_words", "cu_in_words", "raw_words",
               "cu_out_words", "input_words", "truncated", "reused"}],
  "cu_l": str,                  # 最后一个 CU
  "manager_cus": [str],         # manager 实际看到的 CU 列表
  "manager_input_words": int,
  "answer": str,
}
```

参数与规则见 `solution.py` 的 docstring 与题面。
"""


def solve(source, query, worker_out, manager_out, k=8, max_cu=None,
          window=None, instr_words=0, manager_instr_words=0,
          order="left-to-right"):
    raise NotImplementedError
