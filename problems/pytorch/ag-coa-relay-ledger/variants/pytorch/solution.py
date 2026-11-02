"""Chain of Agents：串行接力账（arXiv 2406.02818 Algorithm 1）。

论文协议（正文 §3.1/§3.2 + 附录 B）：

    CU_0 = ""
    for i in 1..l:  CU_i = LLM_{W_i}(I_W, CU_{i-1}, c_i, q)
    answer        = LLM_M(I_M, CU_l, q)

三条必须钉住的协议细节：

1. **只传上一个 CU**。worker i 的输入是 `I_W + CU_{i-1} + c_i + q`，不是
   全部历史 CU；论文脚注 1 明确说「把全部 CU_i 喂给 manager 反而掉分」。
2. **manager 只吃 CU_l**，不吃原文、也不吃中间任何 CU。
3. `CU_0` 是空串，所以第一个 worker 的 `CU_{i-1}` 段占 0 个词。

论文没写的部分按公开实现（rudrankriyam/Chain-of-Agents `utils.py`
`split_into_chunks`）补：先按空行分段、跳过纯空白段、段内空白归一成单
空格、超出窗口的段落硬切成整窗块，剩下的尾巴留作下一块的起点。
"""


def _split_source(text, k):
    """按公开实现的口径把源文本切成词数 <= k 的块。"""
    paragraphs = text.split("\n\n")
    pending, chunks = [], []
    for para in paragraphs:
        if not para.strip():
            continue                      # 纯空白段直接丢掉，不占位
        words = para.split()
        if len(pending) + len(words) > k:
            if pending:                   # 攒着的尾块先落地
                chunks.append(" ".join(pending))
                pending = []
            while len(words) > k:         # 超窗段落硬切，整窗一块
                chunks.append(" ".join(words[:k]))
                words = words[k:]
            pending = words               # 尾巴留作起点（可能是空）
        else:
            pending.extend(words)
    if pending:
        chunks.append(" ".join(pending))
    return chunks


def solve(source, query, worker_out, manager_out, k=8, max_cu=None,
          window=None, instr_words=0, manager_instr_words=0,
          order="left-to-right"):
    """把 CoA 的串行接力摊成账本。

    参数
    ----
    source : str          长文本，按 `k` 个词一块切
    query : str           用户问题，每个 worker 与 manager 都要看到
    worker_out : list[str] 第 i 个 worker 的原始生成（与块一一对应）
    manager_out : str     manager 的最终生成，原样作为 answer
    k : int               agent 窗口（论文默认 8k tokens，这里按词计）
    max_cu : int | None   单个 CU 的词数上限；None 表示不截断
    window : int | None   agent 窗口；给出时才校验 `I_W + CU + c_i + q`
                          塞不塞得进（论文没规定，默认不校验）
    instr_words : int     每个 worker 指令 I_W 占的词数
    manager_instr_words : int  manager 指令 I_M 占的词数
    order : str           `left-to-right` 或 `right-to-left`（论文 §5.5）

    返回账本：`num_chunks` / `chunks` / `order` / `cus` / `workers` /
    `cu_l` / `manager_cus` / `manager_input_words` / `answer`。
    """
    if k < 1:
        raise ValueError("k must be >= 1, got %r" % (k,))
    if max_cu is not None and max_cu < 1:
        raise ValueError("max_cu must be >= 1 or None, got %r" % (max_cu,))
    if order not in ("left-to-right", "right-to-left"):
        raise ValueError("unknown order: %r" % (order,))

    chunks = _split_source(source, k)
    if not chunks:
        raise ValueError("source yields no chunk")
    if len(worker_out) != len(chunks):
        raise ValueError("worker_out has %d entries for %d chunks"
                         % (len(worker_out), len(chunks)))

    # 块在原文里的词序区间：分块把空白吃掉了，只能靠累计词数还原
    spans, pos = [], 0
    for text in chunks:
        n = len(text.split())
        spans.append((pos, pos + n))
        pos += n

    walk = list(range(len(chunks)))
    if order == "right-to-left":
        walk.reverse()                    # 只改处理顺序，块表仍是原文顺序

    q_words = len(query.split())
    workers, cus, cu = [], [], ""
    for idx in walk:
        chunk = chunks[idx]
        cu_words = len(cu.split())
        in_words = instr_words + cu_words + len(chunk.split()) + q_words
        if window is not None and in_words > window:   # 塞不进 agent 窗口
            raise ValueError("window overflow at chunk %d: %d > %d"
                             % (idx, in_words, window))
        raw = worker_out[idx]
        raw_words = len(raw.split())
        if raw_words == 0:
            # 本块没找到相关信息：论文 §3.1 里 W3 的做法是直接重写前一个
            # CU（保留它），不是清空
            out, truncated, reused = cu, False, True
        elif max_cu is not None and raw_words > max_cu:
            out = " ".join(raw.split()[:max_cu])
            truncated, reused = True, False
        else:
            out, truncated, reused = raw, False, False   # 原样返回，不归一空格
        cu = out
        cus.append(cu)
        workers.append({
            "chunk_idx": idx,
            "chunk_words": len(chunk.split()),
            "cu_in_words": cu_words,
            "raw_words": raw_words,
            "cu_out_words": len(cu.split()),
            "input_words": in_words,
            "truncated": truncated,
            "reused": reused,
        })

    manager_words = manager_instr_words + len(cu.split()) + q_words
    if window is not None and manager_words > window:
        raise ValueError("manager window overflow: %d > %d"
                         % (manager_words, window))

    return {
        "num_chunks": len(chunks),
        "chunks": [
            {"idx": i, "text": text, "words": len(text.split()),
             "start": spans[i][0], "end": spans[i][1]}
            for i, text in enumerate(chunks)
        ],
        "order": walk,
        "cus": cus,
        "workers": workers,
        "cu_l": cu,
        # 只有一条：manager 的输入就是最后一个 CU
        "manager_cus": [cu],
        "manager_input_words": manager_words,
        "answer": manager_out,
    }
