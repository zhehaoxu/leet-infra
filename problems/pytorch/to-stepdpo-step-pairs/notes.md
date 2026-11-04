# 解题笔记

## 思路

官方 dvlab-research/Step-DPO 的 data_pipeline 三段账（GPT-4o 调用剥除）：

1. **locate_error_by_gpt4.main**（L26-84）：错误预测筛选账——result 为 False 才送定位；diagram/asy 门、question2cnt 严格 `>` 门控、分片、visited 去重、`"Step 1: "` 补头与 `\n\n` 折叠；
2. **prepare_for_correction.main**（L7-73）：GPT-4 判定输出解析账——decision 段切取、step_num 三段提取链、`Step {n}:` 前缀/续写切分（replace 全局怪癖）；
3. **generate_dataset.main**（L6-67）：偏好对构造账——identifier 分组、首步 `"\nStep "` 切分、chosen 首步去重、Let's think 头的剥/加对称。

上游不入判题域：OpenAI 调用与重试循环、prompt 模板组装（GPT-4 定位提示词本体）、stepdpo_trainer 的损失侧（DPO loss on step pairs，与 `to-dpo-loss-backward` 划界）、eval_math/evaluation 目录的答案抽取与执行器。

与现有题划界：
- **to-dpo-loss-backword / to-dapo-loss**（428 侧）：DPO 的损失与优势账；本题是**数据侧**——偏好对怎么从长 CoT 里构造出来（先错步截断）；
- **782 to-gmpo-geom-mean**：on-policy 策略梯度正向账；本题是 off-policy 偏好数据的离线构造账；
- **rl-* 批（rollout/credit）**：在线交互账；本题纯文本流水线，无奖励模型。

## 关键实现点（官方口径逐条钉死）

- **idx 是 pred_data 下标**：`enumerate(pred_data)` 在 result 过滤之后——visited 文件名 `{idx}.json` 与分片 `idx % n_groups` 都基于过滤后序号，与原始 json 序号错位；
- **门控顺序即语义**：diagram 过滤 → question2cnt 门控+增量 → 分片 → visited。增量在分片/visited 之前意味着被这些检查跳过的项也耗同题名额（visited=[3] fixture 钉死：idx4 因 idx3 被 visited 而变 2>1 skip）；
- **`>` 严格门**：`question2cnt[question] > max_count_per_question`——同题第 2 次（cnt=1）仍放行，第 3 次（cnt=2）才 skip；先查后增；
- **decision 段的 `[-1]` 链**：`split("final decision")[-1].split("summary decision:")[-1]`——两头都认、后者胜；fixture 用「final 段 step 9 + summary 段 step 2」钉 summary 覆盖语义；
- **step_num 提取链**：`split("neutral")[0].split("incorrect")[0]` → `split("\n")[-1]` → `split(";")[-1]` → `split("step")[-1].split(":")[0]` → `int(strip())`。前两步合成的截点是「较早关键词之前」（切分序换位等价，已剔除变异清单）；`count("step") > 1` 拒绝 step 2 **and** step 4 这类歧义段；小数 "2.5" 走 bad_int（int 抛 ValueError，浮点可解析的变异不可行）；
- **completion_full = prefix + completion**：prompt 的 assistant 位是已生成 prefix、completion 字段是**从错误步起的续写**——两者拼合才是完整推理（fixture 把完整推理放两处会翻倍，初版踩坑）；
- **全局 replace 怪癖**：`completion.replace(prefix, "")`——prefix 以 marker 结尾，后文再出现 marker 时整段一起删（`"Step 2: A\nand again Step 2: B"` → `" A\nand again  B"`）；
- **Let's think 的剥/加对称**：prepare 产物剥头存 jsonl（identifier 对齐）、generate_dataset 产出时加回头（训练 prompt 需要）；
- **首步切分 `"\nStep "`**：中间有非 `Step ` 开头的行（sub note）时首步**含**该行——与 `split("\n")[0]` 不同（note 行 fixture 钉死）；rejected 侧同款切分；
- **去重账**：`visited_chosen` 在行循环内重置（每行独立去重）、键是 chosen 首步、`cnt += 1` 在去重检查之前（被去重的尝试也计数）；result=False 的 correction 在分组处就被过滤。

## 测试设计（10 例）

collect 3（basic 分片×门控交互、gates 五路含 question2cnt 语义与 max_total 截断、alpaca style）、locate 4（三种 decision 头 + neutral-only + 透传字段、四类 invalid 含小数 step、全局 replace 怪癖、alpaca style）、pair 2（basic 四条含 rejected note 行、去重键/cnt 计数/result 过滤）、args_validation 1（12 路非法入参）。全部期望值由官方三脚本 exec 沙箱实测钉死（stub openai/jsonlines/tqdm，GPT-4o 调用剥除，文件 IO 落 tmp 后读回验证 {idx}.json 命名与内容）。

## 回源偏差记录

- 官方 locate 的 invalid 是 `continue` 静默丢（四个计数器）；本题返回 `{"valid": False, "reason": ...}` 可观测化（reason 与官方计数器一一对应）。
- 官方三脚本走文件系统（glob 读 json、jsonlines 写 jsonl、{idx}.json 落盘）；本题剥成显式入参/返回值，collect 的 selected 对应官方落盘的 {idx}.json。
- 官方 locate main 的 `while True` API 重试循环与 max_count_total 截断在保存之后；本题剥 API 循环、保留「先保存再截断」的 break 语义。
- `_split_prompt` 在官方三脚本中各复制一份（两 style if/elif）；本题提公共函数，逐条对拍。
