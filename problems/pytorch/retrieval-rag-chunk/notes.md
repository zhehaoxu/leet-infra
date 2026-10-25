# 详解：RAG 重叠滑窗分块

## 出处
LangChain `text_splitter`（`CharacterTextSplitter`/`RecursiveCharacterTextSplitter` 的 `chunk_size`+`chunk_overlap`）、LlamaIndex 窗口/句子滑窗切分。RAG 建库标准预处理：相邻块保留 `overlap` 个重叠 token，缓解「关键句被切断」的召回损失。

## 核心与易错点
1. **步长是 `chunk_size - overlap` 而非 `chunk_size`**：overlap 越大步长越小、块数越多、冗余越高。`overlap=0` 时退化为无重叠硬切分。
2. **成块条件是「起点 < num_tokens」而非「起点+chunk_size ≤ num_tokens」**：末尾会有一块大部分越界、用 `-1` 补——这是常见漏切点（若用 `while start+cs<=n` 会丢掉尾巴块）。
3. **固定宽度 `chunk_size`**：所有行等宽，越界填 `-1` 哨兵，不能返回参差不齐的 list。
4. `step≥1` 由 `overlap<chunk_size` 保证；否则起点不前进会死循环。
5. 下标矩阵只描述「哪些 token 进哪一块」，与真实文本无关。

## 对拍口径
参考解 `starts=arange(0,num_tokens,step)`，广播 `starts[:,None]+arange(cs)[None,:]`，`where(idx<num_tokens, idx, -1)`。判题 `_oracle` 纯 Python：`s=0; while s<num_tokens: 行=[s+j if s+j<num_tokens else -1 for j in range(cs)]; s+=step`（结构不同、逐元素等价）。另设「overlap=0 无重叠硬切」「有效下标覆盖 0..N-1 每个至少一次（重叠不丢 token）」「shape=(块数,chunk_size)、值∈{-1}∪[0,N)」断言。

## 复杂度
\(O(\lceil \text{num\_tokens}/\text{step}\rceil \cdot \text{chunk\_size})\)。
