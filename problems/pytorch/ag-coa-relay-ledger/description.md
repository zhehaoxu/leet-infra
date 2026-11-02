# Chain of Agents 串行接力账

Chain of Agents（CoA，arXiv 2406.02818）处理长文本的方式是「**接力**」而不是「全塞」：
把源文本切成若干块，worker 一个接一个地读自己那块，并顺手把前一个 worker 留下的
交流单元（Communication Unit，CU）改写成新的 CU；最后由 manager 只拿**最后一个 CU**
出答案。

论文与公开实现（`rudrankriyam/Chain-of-Agents`）有一处明显分歧：公开实现的 manager
把**所有** worker 输出一起拼进去，而论文 §3.2 的公式 (2) 是
`Response = LLM_M(I_M, CU_l, q)`，脚注 1 还专门说「把全部 `CU_i` 喂给 manager
反而掉分」。本题按论文口径实现。

本题目把 CoA 剥成纯函数：LLM 调用变成 `worker_out` / `manager_out` 两条入参，
只保留**分块、接力、账本**。实现
`solve(source, query, worker_out, manager_out, k=8, max_cu=None, window=None, instr_words=0, manager_instr_words=0, order="left-to-right")`。

## 输入

- `source`：长文本，按空行（`\n\n`）分段。
- `query`：用户问题，每个 worker 与 manager 都要看到。
- `worker_out`：第 i 个 worker 的原始生成，与块**一一对应**（长度必须等于块数）。
- `manager_out`：manager 的生成，原样作为 `answer`（不 strip）。
- `k`：分块窗口，每块的词数上限。
- `max_cu`：单个 CU 的词数上限；`None` 表示不截断。
- `window`：agent 窗口；给了才校验 `I_W + CU_{i-1} + c_i + q` 塞不塞得进。
- `instr_words` / `manager_instr_words`：`I_W` / `I_M` 各占多少词。
- `order`：`"left-to-right"`（默认）或 `"right-to-left"`。

## 输出

```
{
  "num_chunks": 块数,
  "chunks": [{"idx", "text", "words", "start", "end"}],
  "order": 处理顺序（块下标的列表）,
  "cus": 每轮接力后的 CU 文本（按处理顺序）,
  "workers": [{"chunk_idx", "chunk_words", "cu_in_words", "raw_words",
               "cu_out_words", "input_words", "truncated", "reused"}],
  "cu_l": 最后一个 CU,
  "manager_cus": manager 实际看到的 CU 列表,
  "manager_input_words": manager 输入词数,
  "answer": manager 输出,
}
```

`chunks` 里的 `start` / `end` 是该块在原文里的**词序区间**（左闭右开），与处理顺序无关。

## 分块

按空行切段落，逐段往当前块里塞：

- 纯空白段直接跳过；
- 段内换行、连续空格、首尾空白都会被 `split()` 吃掉，块文本用单空格重连；
- 加上这一段会超 `k` 词时，先把已攒的尾块落地，再处理这一段；
- 单段本身超过 `k` 词时，按整窗硬切成若干块，剩下的尾巴留作下一块的起点；
- 结尾若还攒着词就再落一块；**不产生空块**。

## 接力

`CU_0` 是空串，然后按 `order` 逐块处理：

1. 算本轮输入词数 `I_W + CU_{i-1} + c_i + q`；给了 `window` 且超出就抛错。
2. 取该块的原始生成：词数为 0 视为「本块没有相关信息」，**沿用上一个 CU**
   （论文 §3.1 里 W3 的做法是把答案塞进新 CU 并保留旧信息，不是清空）。
3. 否则若给了 `max_cu` 且超出，按词截断（截断走分词，会归一空格）。
4. 否则原样作为新 CU——**不做任何空白归一**。

## manager 与答案

`manager_cus` 只有一条：`[CU_l]`。`manager_input_words = I_M + |CU_l| + |q|`，
与原文长度、块数、中间 CU 都无关。给了 `window` 且超出同样抛错。`answer` 是
`manager_out` 的**原样**（不剥前后空白）。

## 样例

```
source = "The cat sat.\n\nThe dog ran fast.\n\nA bird flew away."
query  = "who ran", k = 4
worker_out = ["cat", "dog ran", "bird flew"]
返回: num_chunks=3, cus=["cat", "dog ran", "bird flew"],
      cu_l="bird flew", manager_cus=["bird flew"], answer="the dog"
```

## 约束

- 纯函数：不联网、不读写文件、不修改入参。
- `k < 1`、`max_cu` 给了却不是正数、`order` 非法、`worker_out` 条数与块数不符、
  源文本切不出块（空串或全空白），一律抛 `ValueError`。
- 给了 `window` 时，worker 或 manager 的输入超出窗口也抛 `ValueError`。
