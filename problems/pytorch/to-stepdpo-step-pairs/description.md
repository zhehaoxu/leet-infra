# Step-DPO：步级偏好对构造账（错误步定位解析 + 首步截断配对 + 筛选门控）

## 题意

Step-DPO（arXiv:2406.18629, 官方仓 `dvlab-research/Step-DPO`）的核心主张：长 CoT 的错误往往只在某一步，对整条序列做 DPO 会把正确的后续步一起惩罚；正确做法是**从第一步错误处截断**，只对「错误步的续写」构造 chosen/rejected 偏好对。本题取官方 `data_pipeline/` 三段确定性账（GPT-4o 调用剥除，筛选/解析/配对全显式入参）：

```text
mode="collect": items + prompt_style + n_groups/remainder/配额 + visited -> {"selected", "question2cnt", "cnt"}
mode="locate":  gpt4_output + full_prompt + completion + prompt_style -> {"valid", ...}
mode="pair":    corrections + sources + prompt_style -> {"items", "invalid_cnt", "cnt"}
```

## 账本规则（官方逐条钉死）

`mode="collect"`（官方 `locate_error_by_gpt4.main` 的筛选账）：

```text
pred_data = 只收 result 为 False 的项（idx 是过滤后 pred_data 的下标）
question 含 "diagram" 且不含 "asy" -> 跳过（有 Asymptote 代码的图题不跳）
question2cnt 门控：question in dict 且 cnt > max_count_per_question -> 跳过
  （> 严格：同题最多放行 max_count_per_question + 1 次；先查后增；
   增量在分片/visited 检查之前——被跳过的项也耗同题名额）
idx % n_groups != remainder -> 跳过（多进程分片）
idx 在 visited -> 跳过
放行项：completion 补 "Step 1: " 头（原始续写无步标头）；
  incorrect = 补头串的 "\n\n" -> "\n"；folded_solution = gt_output 同款折叠
先保存再检查 cnt >= max_count_total -> break
```

prompt 拆分（`_split_prompt`，两 style）：qwen2-boxed 按
`<|im_start|>user\n` / `\nPlease reason step by step, and put your final answer
within \\boxed{}.<|im_end|>` / `<|im_start|>assistant\n`；alpaca 按
`### Instruction:` / `### Response:`。

`mode="locate"`（官方 `prepare_for_correction.main` 的单条解析账）：

```text
correctness = gpt4_output.lower()
              .split("final decision")[-1].split("summary decision:")[-1].strip()
  （两种 decision 头都认，[-1] 取最后一段——summary 段覆盖 final 段）
不含 neutral/incorrect -> invalid(no_keyword)
step_num 段 = correctness.split("neutral")[0].split("incorrect")[0]
              再 split("\n")[-1].split(";")[-1]
  （先 neutral 后 incorrect 的切分序：保留「较早关键词之前」的文本）
段内 count("step") > 1 -> invalid(ambiguous_step)
split("step")[-1].split(":")[0] 的 int() 失败 -> invalid(bad_int)
completion_full = prompt 拆出的 prefix（已生成部分）+ completion（续写）
不含 "Step {n}:" -> invalid(no_step_marker)（官方静默丢、不计数）
prefix_out = split("Step {n}:")[0] + "Step {n}:"
output = completion_full.replace(prefix_out, "")
  —— replace 全局替换：后文重复出现的 "Step {n}:" 段一起删
返回 prefix 剥 "Let's think step by step.\n"
```

`mode="pair"`（官方 `generate_dataset.main` 的对构造账）：

```text
corrections 按 identifier = question + "||" + 剥 Let's think 的 prefix 分组
  （官方 correction prompt 的 assistant 位带 Let's think 头，剥后与
   prepare 产物 jsonl 行的 prefix 对齐）
source 行 identifier 失配 -> invalid_cnt += 1 跳过
命中行的每条 correction：
  chosen = correction 续写；rejected = source 行的 output（原错误续写）
  首步切分 = split("\nStep ")[0]（"\nStep " 之前的全部，含中间非 Step 行）
  chosen_first_step 已见 -> 只跳过该条（去重键是 chosen 首步；rejected 不去重）
  cnt 计数在去重检查之前（含被去重的尝试）
产出：prefix 加回 "Let's think step by step.\n" 头、original_chosen 全文、
  dataset/type 与 answer 从 source 行透传
```

## 关键怪癖（官方口径）

- collect 的门控顺序：diagram 过滤在 question2cnt 增量**之前**（diagram 项不耗名额），分片/visited 在增量**之后**（这些项耗名额）；
- per-question 门是 `>` 不是 `>=`：同题第 max+1 次仍放行；
- locate 的 `[-1]` 取段语义：GPT-4 输出先 final 段（step 9 判定）再 summary 段（step 2 最终判定）时，summary 段胜出；
- `split(a)[0].split(b)[0]` 保留「较早关键词之前」——neutral 与 incorrect 谁在前都截到第一个坏步（切分序换位是等价变异）；
- output 的全局 replace：completion 尾部重复的 "Step n:" 标头段被一起删（首步即目标步且后文再出现同标头时）；
- pair 的去重只看 chosen 首步且每行重置（visited_chosen 在行循环内初始化）；cnt 跨行累计；
- correction prompt 的 Let's think 头是 identifier 的一部分——不剥就全失配。

## 输入约定

items/corrections/sources 是 dict 列表（缺 result 等字段 KeyError 原样传播）；
invalid 四类返回 `{"valid": False, "reason": ...}` 而非异常（官方静默 skip 的
可观测化）；official `json`/`glob`/`jsonlines` 文件 IO 剥除为显式入参。
