# 解题笔记

## 思路

官方 WENGSYX/Self-Verification 的确定性核心：

1. **main_verifier.py**（L140-259）：候选收集（float 规范化 + 首现去重）、单/多候选分支、verifier 评分累加（L190-208）、argmax 选择与判定；
2. **utils/question_turn_decalrative**（L811-896）：数字挖空子问题构造（数值路径 L835-892）+ declarative 改写缓存（L854-864）。

上游不入判题域：GPT 采样与重生成、answer_cleansing/answer_cleansing_verifier（清洗后串显式入参）、demo 文本、log 与统计（accs/accs_verifier 平均账）。commonsensqa 的问答切分特例（L816-833）与 'Answer Choices' 前缀分支也不入（数值路径只消费全文）。

与现有题划界：
- **rs-self-consistency / rs-ttrl**：多数投票折答案；本题是**内容自洽性评分**选答；
- **775/776**：拼 demo 与选题；本题是单题内部的候选竞争机制；
- **rs-prm-best-of-n**：步级过程奖励；本题是答案级回填验证。

## 关键实现点（官方口径逐条钉死）

- **官方正则 `\d+\.?\d*` 的句点尾**：`"Total is 6."` 抽出 `"6."`（`\.` 可选、`\d*` 零个也行）——挖空匹配以**句段内子串**为准，句段里没有句点 → `"6."` 找不到匹配 → 不替换。跨句点数字（`-2.5`）同理。这两个「挖空不发生」是官方真实行为，不是 bug。
- **number_find 置位缺失**：官方写了 `number_find == False` 条件但从不置 True → 同一数字出现在多个句段时**全部替换**（multi_segment 用例 count('X')==2）。
- **`..` → `.` 修正的来源**：ts 以句号结尾，split 后尾空段重组时多产一个句点（`"6." + "." + "."`）→ 官方 replace 修正。ts-slice 少截一个字符会让这个修正失效（变异点）。
- **`new_question.replace(' .', '.')` 不接收结果**——官方死代码（L878），注释保留；真正的规整是后续 `new_ts.replace(' .', '.')`。
- **declarative 替换是「最后一次出现」**：反向 find 定位 ≡ `rfind`；`"" in 非空串` 恒 True 的语义让空 declarative 分支与 else 分支同落 fallback（m-decl-empty 等价剔除）。
- **评分双向 try float**：判定串与期望值独立转换——"4.0"==4.0 命中、"True" 永不命中数字期望（数字版 verifier 输出数值，demo_B = create_verifier_demo_text 而非 _TF 版）。
- **扁平池 a_idx*3+d**：每候选固定 3 槽（不足 3 个子问题则尾部槽为死槽）——变异 a_idx*2+d 会错位。

## 测试设计（9 例）

- `candidate_collection`：规范化去重（"20"/"5.0"/"5"/空/非数值）+ 单候选路径（scores={0:1}、不构造子问题）+ 单候选错答；
- `subquestion_blank`：挖空结构断言 + 数字版 verifier 计分（["20"]*10 满分 / ["7"] 零分）+ 扁平池死槽 + oracle 对拍；
- `top3_and_numbers`：4 数字取前 3、`"3."` 句点尾抽取、跨句点 `-2.5` 不发生挖空（官方陷阱正面断言）；
- `multi_segment_replace`：同数字双句段全替换（X count==2）+ 重复数字逐条目产生相同子问题 + `"6."` 不替换；
- `declarative_paths`：缓存含 answers[0] 替换最后一次出现 / 不含走 fallback / 空缓存 fallback + oracle 对拍；
- `scoring_and_argmax`：平局取最前（双向）+ 混合判定串（float/字符串双轨）+ 缺失判定池 0 分；
- `comma_and_tail`：千位逗号转句点抽出 `"3.000"` + `'..'` 规整；
- `purity`；`args_validation`（question/采样/判定池三层类型 + GT/declarative 类型 + 消息）。

## 变异收口

32 条设计 → **29 条 CAUGHT + 3 条等价剔除**（m-empty-keep：哨兵即空串；m-decl-empty：空 declarative 经 else 分支同落 fallback；m-val-vo / m-msg-vo：vo 顶层检查被逐元素检查兜底、顶层消息不可达）。
覆盖：规范化丢失/取整/去重删除、declarative 空缓存/未命中/正反定位/偏移、挖空 token/仅首段/双句点规整/ts 截断偏移/前 3 放宽/正则收紧、评分期望不转/判定不转/计数减半/扁平索引错位、argmax 平局反向/单候选清零/单候选路由反转/is_true 用原串/chosen 反转、校验七变异 + 消息一条。

## 工程学备注

- **fixture 设计先跑官方语义再写断言**：跨句点不挖空、`"6."` 尾点抽取、verifier 数字版（非 TF 版）三处都是先调试输出后修正断言——「挖空不发生」比「替换错误」更隐蔽。
- **扁平池索引布局**：2 候选 × 2 子问题占索引 0,1,3,4（索引 2 是死槽）——首轮 v 连续铺 0-3 导致候选 1 全空池，scores {0:9,1:0} 假失败。
- `str(float(x))` 规范化与 `np.argmax` 平局语义是 main_verifier 的两个窄口径，分别被 m-norm-* 与 m-argmax-tie 锁定。
