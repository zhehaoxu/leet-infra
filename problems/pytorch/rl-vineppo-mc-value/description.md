# VinePPO：步级 Monte-Carlo 回报账

## 题面

LLM 多步推理上，PPO 式价值网难以训练：价值头要对每个中间步给出"从这里走
下去最终答对"的期望，监督信号稀疏、方差大。VinePPO（arXiv:2410.01679）的
替代方案是**彻底不要价值网**：在每步分支处续采 k 条完整轨迹，用最终答案正
确性当回报，把每步的 MC 回报均值当 value，advantage 做纯树形差分。本题复
刻官方 `TreeEpisodeUtils` 与 `PathAggregator`/`PathFilter` 全部注册类的确
定性账（torch/Ray/多进程剥除，账目全入参），`solve(mode=...)` 六入口：

1. **score**：树评分。叶判定是 `"answer" in node`（不是"没有 children"）；
   预测答案从 `node["answer"]` 经 `pred_map` 提取——节点自带的 `"pred"`
   字段完全不参与；提取为 `None` → score 0，否则 `int(pred == gold)`；
   `is_correct_answer` 冗余写同值；内部节点 score = 子 score 均值，无子 0。
2. **advantage**：`child.advantage = child.score − parent.score` 的树形差
   分（无自底向上 bias 传播）；叶直接 return 不赋值；root 无 advantage。
3. **paths**：根到叶路径抽取。输入树先 deepcopy；
   `importance_weight = branch_factor ** (max_depth − depth)`，但达
   max_depth 的节点写 1；`repeat_early_stopped_paths` 时**所有浅于
   max_depth 的叶路径**按 importance_weight 份 extend；遍历顺手逐节点
   `pop("children")`；最后 json round-trip。
4. **aggregate**：`none`（原样）/ `unique`（json sort_keys 进 set 去重，
   输出顺序不可复现）/ `top_k_percent`（按各节点
   `advantage.get("advantage", 0)` 的 mean/sum 降序稳定排序取
   `int(len*p)`、`keep_at_least_one` 撑到 1）/ `max_k_paths`
   （`random.Random(42).sample`）。
5. **filter**：`successful`（末节点 `is_correct_answer`，无 "answer" →
   False）/ `non_zero_score`（看 **node_chain[1]** 第一步）/ 三个末节点
   变体。
6. **advantage_split**：字符级优势分摊 + token 清理。字符账
   `ones(len(response_text)) * -777` 哨兵起底，响应文本按首节点
   `"\n"+text+`（多节点再加 `"\n"`）/ 中间 `text+"\n"` / 末 `text` 拼接
   逐段广播赋值；token 账 `ones(len(input_ids)) * -999` 哨兵起底，
   `token advantage = advantage_char[offset_mapping[i][0]]`；断言
   `tokens[0] == "▁"`；清理 = 各自 `[1:]` 剥首位 "▁"。

## 输入/输出

- `score`：`tree, gold, pred_map` → 填充 score 的树
- `advantage`：已评分树 → 填充 advantage 的树
- `paths`：`tree + 4 个抽取参数` → `[{"node_chain": [...]}]`
- `aggregate`：`variant + paths (+ top_k_percent/keep_at_least_one/
  reduction/k)`
- `filter`：`variant + 单条 path` → bool
- `advantage_split`：`node_chain, response_text, input_ids,
  offset_mapping, tokens` → 四元组（全 list）

## 样例

三层树 `root → step1 → {2a(对), 2b → 3b(错)}` + root 下早停叶 shortcut
（answer 不在 pred_map）：

- score：2a=1、3b=0、2b=mean(0)=0、step1=mean(1,0)=0.5、shortcut=0、
  root=mean(0.5,0)=0.25
- advantage：step1=0.25、2a=0.5、2b=−0.5、3b=0、shortcut=−0.25
- paths（bf=2, md=3, repeat_early）：`[3,3,4,2,2,2,2]`——2a（depth 2）
  重复 2 份、shortcut（depth 1）重复 4 份

## 边界与陷阱

- `repeat_early_stopped_paths` 名不副实：正常完成但浅的叶也重复。
- `top_k_percent` 的 advantage 序列含根节点（记 0），mean 分母含它。
- `unique` 的输出顺序是 set 序，测试只能比集合。
- `non_zero_score` 取 `node_chain[1]`，不是末节点。
- 双哨兵：response_text 比 node 拼接长时尾部 −777 残留并传导进 token 账；
  多字符 token 只取起始字符的 advantage。
