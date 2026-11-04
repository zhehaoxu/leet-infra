# VinePPO 题解笔记

## 官方账的关键细节（沙箱实测钉死）

1. **pred 提取只看 `node["answer"]`**：`extract_pred_answer` 用
   `task.extract_predicted_answer_from_text(node["answer"])`，树节点自带
   的 `"pred"` 字段是死数据。用例钉法：把 2a 叶的 `pred` 置 `None`，score
   仍由 answer→pred_map 决定。Shortcut 叶 answer `"The answer is 4."` 不
   在 pred_map → 提取 None → score 0，尽管其 `pred` 是 `"4"`——这是官方
   行为（完整 text 才是 task 的查找键）。
2. **`repeat_early_stopped_paths` 名不副实**：`paths.extend([row] *
   importance_weight)` 的条件是 `max_depth > depth`——任何浅于 max_depth
   的叶（包括正常完成但浅的 2a 叶）都重复。三层树 bf=2 md=3 输出链长
   `[3,3,4,2,2,2,2]`（2a ×2、shortcut ×4）。
3. **importance_weight 的写入值**：`importance_weight if max_depth > depth
   else 1`——达 max_depth 的节点写 1 而不是 bf^0=1 的巧合相同（bf^0 恒为
   1，这里语义上"没有下探余量"）。
4. **`top_k_percent` 的均值分母含根**：`all_advantages = [node.get(
   "advantage", 0) for node in node_chain]`——根无 advantage 记 0 且计入
   分母。
5. **`unique` 输出顺序不可复现**：`{json.dumps(p, sort_keys=True) for p in
   paths}` 是字符串 set，受 PYTHONHASHSEED 影响，测试只能比集合/长度。
6. **`non_zero_score` 取 `node_chain[1]`**：官方假设链至少
   [root, 一步]；语义是"第一步走对了没有"，不是末节点。
7. **字符级 −777 / token 级 −999 双哨兵**：`advantage_char` 起底是
   `ones(len(response_text)) * -777`，node 文本拼接（首节点 `"\n"+text+`
   可选 `"\n"` / 中间 `text+"\n"` / 末 `text`）恰好填满时哨兵不残留；
   response_text 更长则尾部 −777 残留，并经 `offset_mapping[i][0]` 传导
   进 token 账（token 级哨兵 −999 只在 input_ids 比 response_text 长时
   残留）。多字符 token 只取起始字符的 advantage——跨 0.5/1.0 边界的
   token 整个归边界前段。
8. **tokenizer 耦合入参化**：官方 `out_response.input_ids.reshape(-1)` /
   `out_response.tokens()[0] == "▁"` 是 HF BatchEncoding 表面；题里
   input_ids/offset_mapping/tokens 显式传入，`tokens[0]=="▁"` 断言保留。
9. **extract_paths 的副作用链**：deepcopy → dfs 中逐节点
   `pop("children", None)`（官方注释：序列化提速）→ `json.loads(json.
   dumps(paths))` round-trip（元组变列表、int/float 保真）。

## 与相邻题的划界

- `to-gae`：GAE 有可学习价值网 + TD 残差 + λ 折扣；本题 value 就是 MC
  回报均值、无折扣、无网。
- `to-ppo-value-clip`：那题是 value 头的 clip 损失；本题根本没有可学
  value。
- `to-grpo-advantage`：GRPO 是组级归一化 advantage；本题是树形差分。

## 沙箱搭建教训

- 抽取 `convert_path_to_episode` 的 advantage 段时，**先剥缩进、后拼
  return**——return 行（0 缩进）参与 min-indent 计算会把 bind 顶成 0，
  函数体 12 空格缩进没剥掉，exec 时 IndentationError。
- Registrable stub 必须 **per-base-class 注册表**（`cls.__dict__` 分层），
  否则 PathFilter 的 `"none"` 注册会覆盖 PathAggregator 的 `"none"`。
- 类体直接 exec 不加缩进前缀——加了会变嵌套类。
- fixture 三处坑：multichar token 组最初断言了 enc2（全字符 token）的
  输出（out2/out3 错位）且 offsets 全在 0.5 段内没有区分度——改为
  `[30,36)` 跨边界 + `[34,37)` 边界起点两组；补了 −777/−999 哨兵残留组；
  `paths_repeat_early` 在 dump 里是 json.dumps 字符串，test 里要再
  `json.loads`（或 json.dumps 后比较）。
