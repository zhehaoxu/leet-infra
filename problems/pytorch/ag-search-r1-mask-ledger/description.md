# Search-R1 检索掩码账（ag-search-r1-mask-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：730

把 Search-R1（arXiv **2503.09516**）多轮 rollout 的「生成 → 检索 → 观测」序列账无状态化：给定一批行的 prompt、逐次生成 token、检索结果与观测包装 token，复算每轮的**截断与动作解析**、观测段的**掩码**、训练侧（前缀锚定）与上下文侧（后缀锚定）**两个窗口**，以及 rollout 统计。回源 [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) main：`search_r1/llm_agent/generation.py` 的 `_postprocess_responses`、`postprocess_predictions`、`execute_predictions`、`run_llm_loop`、`_update_right_side`、`_update_rolling_state`、`_compose_final_output`，以及 `search_r1/llm_agent/tensor_helper.py`。

## 你要实现的

`solve(rows, vocab, search_results=None, wraps=None, max_turns=3, max_start_length=512, max_prompt_length=64, max_obs_length=32, drop_toks=("<|im_end|>", "<|endoftext|>"))`：返回整批 rollout 的账。

- `rows`：`[{"prompt": [id, ...], "gens": [[id, ...], ...]}]`。`gens` 按顺序被消费——主循环每轮取一次，若主循环用尽仍有活跃行，再取一次做 final rollout。
- `vocab`：`{id: token 文本}`，即显式 tokenizer（四个标签 `<search>`/`</search>`/`<answer>`/`</answer>` 各占一个 token）。
- `search_results`：`{query 文本: [结果 token id]}`（官方在 `execute_predictions` 里对结果做了 `.strip()`，这里直接给 strip 后的结果 token）。
- `wraps`：`{"obs_prefix": [...], "obs_suffix": [...], "retry": [...]}`——观测串 `"\n\n<information>" + 结果 + "</information>\n\n"` 的前后包装 token，以及非法动作的固定重试提示 token。

返回 `{"rows": [...], "active_num_list": [...]}`。每行账：`actions` / `contents` / `obs_kinds` / `obs_truncated` / `turns` / `valid_actions` / `valid_searches` / `finished` / `right_ids` / `segments` / `dropped` / `final_ids` / `final_info_mask` / `rolling_ids`。

**账目只记真实 token**：左端 prompt pad 与右端补偿 pad 在 `attention_mask` / `info_mask` 中恒 0，因此不入账（见 notes 的推导）。

## 账点（源码逐行核对）

1. **截断：`</search>` 优先，且与位置无关**。`_postprocess_responses` 是 `resp.split('</search>')[0] + '</search>' if '</search>' in resp else resp.split('</answer>')[0] + '</answer>' if '</answer>' in resp else resp`——先问全文有没有 `</search>`。所以「先答后搜」的生成被截到**最后一个 search 闭标签**为止，search 段整段留在 policy token 里（动作却是 answer）；「先搜后答」则把后面全部砍掉。
2. **解析要配对，且按位置优先**。`postprocess_predictions` 用 `re.search(r'<(search|answer)>(.*?)</\1>', prediction, re.DOTALL)`：**带同名闭标签的第一个开标签**胜出，内容取其后**最近**的闭标签之间再 `.strip()`。只有闭标签（`</search>` 没有对应 `<search>`）→ 动作 `None`；同名嵌套（`<search>a<search>b</search>`）→ 内容是 `"a<search>b"`。截断与解析的口径不同，二者可以打架（第 1 条）。
3. **特殊 token 整段消失**。`batch_decode(..., skip_special_tokens=True)` 之后是**重新 tokenize**：`<|im_end|>`/`<|endoftext|>` 不但不进 policy 段，连 token 数都少了一个（少一个 loss 位），而且截断/解析都按**丢弃后**的序列做（丢掉中间的特殊 token 会让两个标签变成相邻）。
4. **非法动作照样吃掉一轮**。`execute_predictions` 的 else 分支塞固定重试提示、`dones=0`、`valid_action=0`；源码里三个反斜杠续行**不注入换行**（文本是单空格连接）。这一轮的 policy token 仍参与 loss。
5. **观测截前缀**。`_process_next_obs` 里 `next_obs_ids[:, :self.config.max_obs_length]`：保留**开头**，`</information>` 常被砍掉；重试提示同样受这个上限约束（超长时每轮都被砍成同一段前缀）。
6. **训练侧是前缀锚定**。`_update_right_side` 算出 `effective_len` 后 `max_len = min(max_prompt_length, effective_len)`，再 `responses[:, :max_len]`——保留**最老**的 token；攒满后新的 policy/观测 token **静默丢弃**，但动作照常生效、行照常完成（`dropped` 记被丢的个数）。因为 `own ≤ eff`，`min(own, min(P, eff)) = min(own, P)`，所以逐行算就是精确的（证明见 notes）。
7. **上下文侧是后缀锚定**。`_update_rolling_state` 用 `[:, -max_len:]` 保留**最新**的窗口——于是训练序列是「prompt 尾段 + 最老的 P 个生成」，上下文是「最新的 P 个生成」。常见的结局是：答案进了上下文、却因为封顶没进 loss（`rolling_ids != right_ids`）。
8. **左端 prompt 也是后缀截断**。`initial_input_ids[:, -max_start_length:]` 取 prompt **尾部**；`max_start_length=0` 时 `prompt[-0:] == prompt[0:]` 是**全量**而非空。
9. **`max_prompt_length=0` 两侧语义相反**：右端 `[:0]` 是空序列，上下文 `[-0:]` 是全量。同一个 0 在两侧给出相反结果。
10. **final rollout 是「不检索」的补一刀**。主循环之后若仍有活跃行，再生成一次且 `do_search=False`：`next_obs` 整个丢弃（不追加观测）、**不更新 `turns_stats`**；这轮的 search 仍算 `valid_action`/`is_search`，但 `dones=0` → 该行永远不算完成。所以 `turns` 可以等于 `max_turns + 1`，而 `finished` 仍是 False。
11. **统计口径**：`turns_stats` 从 1 起、每步只给「仍未完成」的行 +1 ⇒ 它等于该行实际**生成次数**；`valid_action_stats`/`valid_search_stats` 在主循环与 final rollout 都累加；`active_num_list` = 初始活跃数 + 每步之后 + （若还有活跃行）final 之后，**全员完成时提前 break**（不多出尾项）。
12. **`info_mask` 的口径**：`_compose_final_output` 里 `info_mask = attention(prompt) ⊕ attention(responses_with_info_mask)`，观测段在 `responses_with_info_mask` 侧被替换成 pad，所以 mask = prompt 段 1、policy 段 1、观测段 0。域约束（ValueError）：id 不在 `vocab`、某轮真被消费却缺生成、search 的 query 在 `search_results` 里没有结果、任一边界为负。

## 样例

```python
V = {1: "<search>", 2: "</search>", 3: "<answer>", 4: "</answer>",
     5: "<|im_end|>", 7: "\n\n", 8: "<information>",
     9: "</information>\n\n", 11: "RETRY", 20: "Q",
     30: "weather", 31: "42", 40: "Doc1", 41: " Doc2"}
W = {"obs_prefix": [7, 8], "obs_suffix": [9], "retry": [11]}
rows = [{"prompt": [20], "gens": [[1, 30, 2], [3, 31, 4]]}]
led = solve(rows, V, {"weather": [40, 41]}, W, max_turns=3)
r = led["rows"][0]
# actions == ["search", "answer"]；contents == ["weather", "42"]
# turns == 2；valid_searches == 1；finished == True
# segments == [["policy", 3], ["obs", 5], ["policy", 3]]
# final_info_mask == [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
# active_num_list == [1, 1, 0]（第 2 步答完即 break，无 final rollout）
```

## 计分

15 个检查点：截断的 search 优先、解析的配对与最近闭标签、非法动作的重试轮、观测截前缀（含重试提示）、final rollout 的三条语义、训练侧封顶与 `dropped`、两个窗口的分歧、prompt 尾段截断与 `0` 的切片语义、`max_prompt_length=0` 的双面性、特殊 token 丢弃、`active_num_list` 形状、四个默认值、域约束，以及 60 组随机 rollout 对拍。
