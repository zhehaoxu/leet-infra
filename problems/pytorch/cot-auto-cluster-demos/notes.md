# 解题笔记

## 思路

官方 amazon-science/auto-cot 的 run_demo.py（L41-154），取 sampling=center 确定性路径：

1. **输入侧**：官方从 zero-shot-CoT 的 log 文件解析候选池（Q:/A:/pred_mode/GT : 行协议 + Therefore 触发 rationale 定稿的状态机）——解析层不入判题域，池直接作入参；SentenceTransformer 编码、KMeans 拟合与 PCA 可视化同不入（`retrieval-kmeans-lloyd` 占 k-means 算子），`labels_` 与 `transform` 距离矩阵显式入参；
2. **判题域**：分簇 → 簇内距心稳定升序 → 逐候选五重过滤 → 每簇一个 break → rationale 清洗组装。

与现有题划界：
- **rs-self-consistency**：多条候选折成一个答案（投票），本题是选代表拼 prompt（无聚合）；
- **retrieval-kmeans-lloyd**：通用 k-means 算子，本题消费其输出（assignment + dist）；
- 后续 cot 线（②active-prompt ③self-verify ④RoT ⑤cumulative）共用本题的「簇选拼 demo」骨架。

## 关键实现点

- **稳定排序即 tie-break**：分簇 append 按 sid 升序 + stable sort by dist，平局自动保持 sid 序——显式 `(dist, sid)` 双键与官方单键 stable 等价（变异 m-tie-key 因此剔除）。
- **清洗链的冗余**：`replace("\n\n","\n").replace("\n"," ")` 后接 `" ".join(split())`——`split()` 按任意空白切，replace 链不改 token 序列，最终等价 `" ".join(r.split())`。保留官方写法（忠实度），变异 m-clean-replace 剔除。
- **[-2] 的安全性链**：条件 c（endswith "."）保证 `r.split(".")` 至少两段，[-2] 不会越界——过滤顺序是语义的一部分，变异 b/c 与 e 的相对顺序可被抓。
- **[-2] 与尾 10 词的关系**：[-2] 永远是最后一个完整句子（结尾句点保证末段为空串）；尾 10 词 ⊆ 最后一句（若 ≥10 词）。「[-2] 命中而尾 10 不命中」要求最后一句 ≥10 词且答案在句首——fixture s0 用 18 词长句首词答案构造。
- **算术回验大小写敏感**：`in` 是字面子串匹配，"twelve" 不命中 "Twelve"——fixture 与变异用例都要注意（本轮踩过两次）。
- **死数据列**：dist 的非所属列官方不读，但读错列（m-dist-col）改变簇内排序——fixture 里簇 1 双候选 s6/s9 且 [2] 列故意反转（s6=0.1 < s9=9.9）使错列变异分化。

## 测试设计（9 例）

- `main_oracle`：10 样本 k=5（含空簇）双 max_ra_len 路径全字段对拍；s2 距心最近（0.5）但五行 rationale——max_ra_len=2 时让位 s0、=5 时独占簇 0（选择顺序随上限变化）；
- `gold_ans_passthrough`：带空格金标 strip 透传 + None 路径；
- `word_count_filter`：55 词通过、61 词被拦；
- `ra_len_boundary`：行数恰等上限通过、超一行不过；\n\n 折叠计数；
- `arithmetic_verify`：四簇四路径（[-2] 命中 / 尾 10 词命中 / 答案不在文本 / arithmetic=False 放行）；
- `tie_and_empty`：距离平局取小下标 + 空簇无 demo；
- `folding_cleanup`：清洗链精确断言 + 行数按 \n 计；
- `purity`：输入不被原地修改；
- `args_validation`：类型/等长/越界/bool/行宽/数值/上限/消息，含 tuple 行与非数值元素。

## 变异收口

33 条设计 → **32 条 CAUGHT + 1 等价剔除**（m-val-k0：与 questions 非空检查互为兜底，单独删除不可观测）：
- 分簇错列/assignment 用下标/排序反转；过滤五条各删改 + e 的 [-1]/尾 3 词 + e 翻转；break 删除/行数过滤改 break（簇级放弃）；默认参数漂移；清洗不折叠/gold 不 strip/gold 恒 None/簇数少报/selected 与 cluster_order 错位；校验十条（名称类型/等长/bool 混入/越界/行宽/数值/上限含 0/bool 上限/arithmetic 类型/gold 等长）+ 消息两条。

## 工程学备注

- **变异脚本的 fatal 判据**：变异导致 crash 时 judge 返回 fatal（passed=0/total=0），`passed < total` 判据会误读 SURVIVED——用 `not accepted or fatal`（771 教训的推广版）。
- **fixture 演进要跑全用例**：s2 从三行改五行、s0 改长句、簇 1 加 s9——每步改动都让 main_oracle 的两个 max_ra_len 路径、gold、purity 连锁重验（本轮 main_oracle 反复失败五轮全因 fixture 与断言不同步）。
- **break 的每簇语义**：断言 max_ra_len=5 的 selected 时误写 [2,0,9,7,8]（s2 通过后 s0 仍收）——break 后簇 0 只有 s2，[2,9,7,8]。
