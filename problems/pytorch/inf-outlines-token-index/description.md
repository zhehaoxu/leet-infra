# 正则编译产物：token 级 Index 构建账

受限解码的运行期必须对每个解码步 O(1) 回答「当前状态下哪些 token 合法、走到哪」，所以 outlines-core 把这笔账在**编译期**一次算清：把 byte 级 DFA 与词表交叉，压缩成一张 `state -> token_id -> next_state` 的索引（`Index`，源码 `dottxt-ai/outlines-core` 的 `src/index.rs`）。本题要求从给定的 byte 级 DFA（state 0 为 anchored 起点，未写出的字节落到 sink）出发，复现这条构建算法的全部细节账。

## 流程

1. **枚举**：从 state 0 出发，用 LIFO 栈（`stack.pop()`）逐个处理 state；token 按**词表插入顺序**遍历。
2. **单 token 推进**：token 按 UTF-8 字节逐个走 DFA；任一字节落到 sink_state 即整个 token 作废——哪怕前缀已合法。token 可以停在非 match 的中段 state（只要该 state 有 DFA 出边，即"有用"）。
3. **有用性过滤**：到达的 next_state 只有在「是 match state」或「存在某个字节使它不落 sink」时才收录：把该 token 的**全部 id**（一词多 id）加边 `current ->(id)-> next_state`，并把 next_state 入栈（未见过时）。
4. **final 判定**：被枚举到的 match state 才是 final state；从未被 token 到达、也不在 state 0 的 match state 不算。
5. **eos 自环**：全部枚举结束后，给每个 final state 追加 `eos_token_id -> 自身`。eos 只出现在这个自环里，绝不作为普通 token 边出现；含 eos id 的 token 整个跳过。
6. **不兼容报错**：某 state 非 match 且没有任何 token 成功加边 → 词表与正则不兼容，立即返回报错 dict：出错 state 编号 + `missing_tokens`（该 state 下所有不落 sink 的字节按升序，`<128` 输出 ASCII 字符、否则输出 `\xHH` 两位小写十六进制）。match state 即使没有 token 出边也不报错。

## 输入格式

- `transitions`：`dict[state -> dict[byte -> next]]`，byte 取 0..255；未写出的字节转移到 `sink_state`；整个 state 缺省视为全落 sink。
- `match_states`：DFA 接受态集合；`sink_state`：死态编号（dead/quit 合并抽象）；`vocab`：`dict[token_str -> list[token_id]]`（插入序即遍历序）；`eos_token_id`。

## 任务

实现 `solve(transitions, match_states, sink_state, vocab, eos_token_id)`，返回：

```python
{
  "initial_state": 0,
  "final_states": [...],      # 升序
  "transitions": {...},       # 含 eos 自环
  "incompatible": None,       # 或 {"state": s, "missing_tokens": [...]}
}
```

## 例子

```
输入: transitions = {0: {ord('a'): 1}, 1: {ord('b'): 2}},
      match_states = {2}, sink_state = 3,
      vocab = {"a": [0], "b": [1]}, eos_token_id = 4
输出: final_states = [2],
      transitions = {0: {0: 1}, 1: {1: 2}, 2: {4: 2}},
      incompatible = None
```

## 约束

- DFA 的 state 编号非负且含 0；token 为非空字符串；`eos_token_id` 可能同时出现在某 token 的 id 列表里（此时该 token 整体跳过）。
- 报错发生时官方直接 Err，整个 Index 不可观察——只需返回报错 dict。
