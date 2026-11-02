# notes（ag-search-r1-mask-ledger）

## 回源

- 论文 arXiv **2503.09516**（Search-R1，2025）§3：多轮 rollout 里模型交替产生 `<search>query</search>` 与 `<answer>ans</answer>`，检索结果以 `<information>…</information>` 拼回上下文；训练用 outcome reward，所以**必须把观测段从 loss 里剔掉**（`info_mask`），否则模型会去拟合检索器返回的文档。
- 代码：`PeterGriffinJin/Search-R1` main（blob `6b68cb003ac3f943d45eb8d5cf48a7ebee5cd1f6`，470 行）的 `search_r1/llm_agent/generation.py`；`tensor_helper.py`（`15a7c7c084c4f952533f43b214f987db81075255`，75 行，`cut_to_effective_len` / `convert_pad_structure` / `create_attention_mask` / `create_position_ids` / `_example_level_pad`）。抓码走 GitHub blob API + base64 拿原字节（`raw.githubusercontent` 偶发 SIGKILL、WebFetch 会吞缩进）。
- 上游是 verl：`run_llm_loop` 是 `verl/trainer/ppo/ray_trainer.py` 里 rollout 的改写，`DataProto` 的 batch 是 rank-2 张量（左 pad 的 prompt、右 pad 的 response）。本题把张量层剥掉，只留**逐行 token 账**。

## 官方 quirk（逐条核实）

1. **截断优先级与位置无关**：`_postprocess_responses` 先判 `'</search>' in resp`，再判 `'</answer>'`，两者都用 `split(...)[0] + 闭标签`。所以「先答后搜」的生成会保留整段 search（动作却是 answer）；「先搜后答」把 answer 全砍。这两条在 `check_search_priority_cut` 里各钉一次。
2. **解析是 `re.search` 的语义，不是「第一个闭标签」**：`<(search|answer)>(.*?)</\1>` + DOTALL。开标签没有同名闭标签时正则回溯换下一个开标签，故 `</search>` 而无 `<search>` 时动作是 `None`（走重试），不是 search。同名嵌套取**最近**闭标签。
3. **重新 tokenize 会吃掉特殊 token**：`batch_decode(skip_special_tokens=True)` → 字符串 → `_batch_tokenize`。`<|im_end|>` 既不在 policy 段、也不占位数。这与「保留原始 id 再置掩码」是两种口径，本题按官方（重新 tokenize）。
4. **重试提示是单空格连成的**：源码
   ```python
   next_obs.append(f'\nMy previous action is invalid. \
   If I want to search, ...')
   ```
   三个反斜杠续行**不注入换行**，续行首列无缩进 → 文本是 `"...invalid. If I want to search, ..."`。本题把这段做成 `wraps["retry"]` 入参，只考「它占多少 token、掩 0」。
5. **观测截前缀**：`_process_next_obs` 的 `next_obs_ids[:, :max_obs_length]`。真实配置里 `max_obs_length=500`、每条观测 3 篇文档，超长时尾部（含 `</information>`）被砍——这是训练里真实存在的破格式现象。重试提示也走同一函数。
6. **训练侧前缀锚定、上下文后缀锚定**：`_update_right_side` 用 `[:, :max_len]`、`_update_rolling_state` 用 `[:, -max_len:]`。同一 rollout 的两个窗口因此可以完全不同：封顶之后答案只进上下文、不进 loss。`check_two_windows` 用 `max_prompt_length=6` 把两者钉成不同的 6 元组。
7. **`min` 折叠**：`max_len = min(P, eff)`，`eff = max_j own_j`，逐行保留 `min(own, max_len)`。因 `own ≤ eff`，`min(own, min(P, eff)) = min(own, P, eff) = min(own, P)`——**批量最大值不影响任何一行**，所以本题可以逐行算（这是保真性证明，不是等价变异）。
8. **`0` 的两面性**：`max_prompt_length=0` 让右端 `[:0]` 为空、上下文 `[-0:]` 为全量；`max_start_length=0` 让 `prompt[-0:]` 等于 `prompt[0:]`（全量）。两处都在 `check_zero_prompt_length` / `check_prompt_tail` 里钉死。
9. **final rollout 三条语义**：`do_search=False` 时 `search_results` 退化成 `[''] * n`，观测 `'\n\n<information></information>\n\n'` 被**丢弃**（`_` 接），`turns_stats` 不更新，`dones=0`。所以「最后一轮又搜了一次」的行**永远不算完成**，`turns` 可以等于 `max_turns + 1`。
10. **`turns_stats` 恒等于生成次数**：从 `torch.ones` 起、每步只给「仍未完成」的行 +1；主循环 k 步 + final 一次 = k+1 次生成，而完成于第 k' 步的行恰得 k'。`check_active_num_list` 用一个 3 行混合批（答完 / 一直搜 / 一直非法）钉住 `[3, 2, 2, 2]`。
11. **`if next_obs_ids != None:` 是张量比较**：实测（torch 2.x）`tensor != None` 走 `__ne__` 返回 `NotImplemented`、由 Python 回落到身份比较 → `True`。所以这行等价于 `is not None`；answer 行传进来的是**全 pad 行**（`padding='longest'` 把空串补齐），对账的贡献是 0 个真实 token。
12. **pad 尾列存在但不入账**：已完成的行走 `_example_pad` 每步仍拼一个 pad 块，`[:, :max_len]` 之后可能留 `cap - content_len` 个尾 pad。它们在 `attention_mask` / `info_mask` 里恒 0，且 `(cumsum-1)*attn` 让它们的 position_id 也回到 0——所以本题只记真实 token，把这一层显式排除（并在 description 里写明）。

## 出题工程

- 首轮 `check_default_args` 断言全部通过但我自己写错了 oracle：`left = prompt[len(prompt) - S:]` 在 `S > len(prompt)` 时给出**负下标窗口**（300 token 的 prompt + S=512 → 只留 212 个），而官方 `prompt[-512:]` 是全量。改成 `S == 0 or S >= len` 就整段。这条 bug 反而变成考点（`check_prompt_tail` 同时钉 `S=0`/`S=1`/`S=512`）。
- 相邻同类段必须合并：final rollout 的 policy 直接接在主循环最后一个 policy 后面（中间没有观测），不合并会出现 `[["policy",4],["policy",1],["policy",2]]` 这种分段。oracle 是「(kind,id) 展平后按连续同类压缩」，solution 是「攒的时候合并」，两条独立路径对上才说明分段口径唯一。
- `check_errors` **必须直接调 solve**：第一版通过 `_led`（内部同时调 oracle 与 solve），参考解自己抛的 `ValueError("negative bound")` 把被测实现的空边界检查兜住了 → `m-bound-no-check` SURVIVED。改成直调 solve 后立刻 CAUGHT。这是本类「域约束题」的通用坑。
- 用例侧边角：`decode_drop` 的第二个场景要补第二次生成（丢掉 `<|im_end|>` 后 `<search>` 仍是动作，行不会结束）；`errors` 的「缺生成」要用会继续的行（一直 search），否则答完即 break、根本不消费第二轮。
- 默认值检查点特意造了三个长输入：23 token 的观测（抓 `max_obs_length=16`）、42 token 的生成（抓 `max_prompt_length=32`）、300 token 的 prompt（抓 `max_start_length=256`），外加 `turns=4`（抓 `max_turns=4`）。四个默认值都被单独打出来。
- 变异 47 条**全部 CAUGHT，0 条等价**：截断 4、解析 3、特殊 token 2、观测 4（含 off-by-one）、轮次/统计 6、窗口 5、掩码 3、分段 2、域约束 3、默认值 5、其它 10。没有出现「不可观测中间量」，因为这个题面本身就是把中间量（`segments`/`dropped`/`obs_truncated`/`rolling_ids`）全部暴露出来的账。
- 随机对拍 60 组：随机构型（`max_turns 0..3`、`S ∈ {0,1,2,512}`、`P ∈ {0,3,8,64}`、`O ∈ {0,1,4,32}`）+ 随机生成（含未闭合标签、特殊 token、零长生成）；`search_results` 用 `__contains__` 恒真的 dict 子类兜住随机 query（`deepcopy` 后行为不变）。
