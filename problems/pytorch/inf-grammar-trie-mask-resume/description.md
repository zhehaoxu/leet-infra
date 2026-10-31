# 受限解码 token 前缀 trie 与增量 mask 复用账

## 题意

xgrammar（mlc-ai/xgrammar）为语法约束解码做的两件预处理/增量事：`TokenizerInfo` 把 vocab 组织成「排序数组 + 子树区间」的伪 trie（`sorted_decoded_vocab` + `trie_subtree_nodes_range`），`GrammarMatcher::FillBitmaskForStates` 在按字典序遍历 uncertain token 时用 LCP 三分支做前缀增量复用。本题逐层转写这套账，并把 Earley 状态机抽象为确定性的「目标串前缀闭包」byte 状态机。与 `inf-grammar-mask-dfa` 划界：那是单步 DFA 状态到 token 的掩码判定，本题是**跨 token 的前缀复用与子树整段跳过**。

## 求解

**第一层（TokenizerInfo 账）**：
1. 分类：token 命中官方检测表 `DETECTION_STOP_TOKENS` 8 条（`</s>`、`<|end_of_text|>`、`<|eot_id|>`、`<|endoftext|>`、`<eos>`、`<|eos|>`、`<end_of_turn>`、`<｜end▁of▁sentence｜>`）或显式 `stop_token_ids` → stop（显式模式下**不用**检测表）；`token == ""` → special；其余进 sorted_decoded_vocab。
2. 排序：按 token 字符串字典序；并列钉死按 token id 升序（官方 `std::sort` 不稳定，题面钉死 stable 口径）。
3. `trie_subtree_nodes_range[i]`（值语义：子树 = `[i, range[i])`）：官方单调栈逐行转写，弹栈条件是 `cur.find(top) == npos`——**子串包含判断而非前缀判断，官方 quirk 原样保留**（栈顶 `xab` 遇 `xxab`：`find("xab")==1≠npos` 不弹，range 被错误延长到 [2,2)）。

**第二层（增量复用账）**：uncertain 按 sorted 下标升序遍历（入参 token id，内部转升序）：

- **subtree-skip**：`cur_idx < last_rejected_range` → 整段子树直接拒（官方 kRejected delta）；
- **lcp 短路**：与 prev_token 的最长公共前缀 `lcp > prev_matched`（prev 匹配失败位置之后仍共享前缀）→ 状态机确定性 ⇒ 同一字节处必失败 → 拒并置 `last_rejected_range = subtree_range[cur]`；
- **回退续推**：`lcp < prev_matched` → `prev_matched = lcp`（官方 `PopLastStates`），从公共前缀续推；`lcp == prev_matched` 原地续推；
- walk 从 `prev_matched` 起逐字节 advance（`advance(s)` ⇔ s 是 targets 中某串的前缀），失败记 walk-rejected、置 range，`prev_matched` 停在最后成功长度（失败不回退，供下一 token 短路）；
- prev_token **每轮更新**（接受/拒绝都更新；官方 atomic-retry 置 nullptr 的路径不入题）。

## 示例

```
输入：vocab = ["xab","xxab"]，targets = ["xa","xxab"]，uncertain = [0,1]
输出：trie_subtree_nodes_range = [2,2]（quirk：正确语义应为 [1,2]）；
     ledger = [walk-rejected, subtree-skip]，accepted = []
     （xxab 本可独立 walk 接受，被 quirk 延长的子树整段过拒）
```

## 约束

- encoded_vocab/targets 为 str 列表；uncertain_token_ids 为 int 列表，越界、含 stop/special、bool → `ValueError`；stop_token_ids 越界/bool → `ValueError`。
- 返回 sorted_token_ids / sorted_tokens / trie_subtree_nodes_range / stop_token_ids / special_token_ids / accepted_token_ids / ledger（每 token 的 branch ∈ accepted | walk-rejected | lcp-short-circuit | subtree-skip，lcp 与 matched_from 账）。
