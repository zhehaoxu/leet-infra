# SWEET-RL 实现笔记（792）

回源：facebookresearch/sweet_rl `scripts/` 四脚本 + 论文 arXiv:2503.15478 §"SWEET-RL"。训练在配套 fork YifeiZhou02/collab_openrlhf（openrlhf 的 fork，--mean_log_prob 支持 length-normalized DPO）。沙箱 oracle：官方逻辑直译 + `random.Random(42)` 逐对对拍（/tmp/sweet_check.py）。

## 逐入口口径

### judge_prompt（rank_best_of_n.py::formatting+main prompt 段）

- GT 分支：`"In light that the final answer is: {ground_truth}." + input + output + RESPONSE_TEMPLATE`；no_use_ground_truth 分支无前缀。
- RESPONSE_TEMPLATE = `"<|start_header_id|>assistant<|end_header_id|>\n\n"`（Llama-3.1 chat 模板）。前缀与 input 之间**无空格**（f-string 直拼）。

### judge_score（rank_best_of_n.py::main 打分段）

四步账，顺序钉死：
1. `prompt_logprobs[:-4]` 截尾——官方注释 'the last four digits are another empty response'。
2. 循环内 `if prompt_logprob is not None:` 才 append——logprob 与 decoded_token **同步**收集（None 位两者都不进）。
3. `find_last_index(all_tokens, eos_token)`——**最后一个** eos（`len(lst)-1-lst[::-1].index(element)`，无匹配返回 -1）。
4. `tail = logprobs[last+1:]`；use_sum → `sum(tail)`，否则 `sum(tail)/len(tail)`（mean = length normalization，默认）。
- 每项 prompt_logprob 是 `{token_id: Logprob}` dict（prompt_logprobs=0 采样时单项）；官方遍历 `.values()`（本题入参化为 (logprob, token) 元组）。
- tail 为空时 ZeroDivisionError——真实数据 eos 后至少 4 个截尾位保证非空（见 judge_score fixture 设计）。

### rank_against_reference（rank_best_of_n.py::main 重排段）

- `judge_results - reference_judge_results`，两边 reshape(-1, best_of_n)；reference 来自 sample_best_of_n 用**旧策略**算的 additional_reference_logprobs（mean 对应 judge mean、sum 对应 sum）。
- `torch.argsort(descending=True, dim=1)` → 本题 `np.argsort(-adjusted, axis=1)`（稳定序一致：相同分数按下标升序，test 的 [0,1,3,2] 锚钉住）。
- 官方把 judge_result 写回 data 供 generate_random_pairs 消费。

### flatten（sample_best_of_n.py::main 展平段）

- `for i, dh in enumerate(dialogue_history): if "input" in dh:` ——只展开带 input 的条目（human 模拟器的发言；agent 的纯 output 条目跳过）。
- 每条 {input, input_with_ground_truth（GT 前缀+input）, older_output, reward, ground_truth, messages[:i+1]}——messages 截到当前 turn；本题保留 input/older_output/messages 三键 + input_with_ground_truth。

### pairs（generate_random_pairs_from_ranks.py::main）

- `random.seed(42)` 是**脚本级全局**种子；本题入参 `random.Random(seed)`，seed=42 时抽签序列与官方一致（沙箱逐对对拍：choice 先 chosen 后 rejected、最后 shuffle）。
- `extract_indices`：`np.argsort(judge_results)` **升序**；`chosen_index = random.choice(range(n//2))`、`rejected_index = random.choice(range(n//2, n))`——按**位置**挑，返回 `argsorted_indices[位置]`（排序后的原始下标）。然后主循环按**原始分数**交换保证 chosen > rejected——净效果：chosen 是「前半位置挑的」与「后半位置挑的」中较高分者，**通常不是绝对最优**（bestworst 变体才是 argsort 首尾）。
- num_pairs 次重复抽签（同一 judge_result 反复用，rng 状态延续）。
- get_sft：`random.sample(range(n), 2)` 后 **chosen=rejected=indices[0]**（官方原样怪癖——SFT 只需要正样本）；文本恒 output+eos 且带 prompt。
- no_prompt：`input + output + eos`（上下文进 response 侧）；否则 `output + eos` + prompt 分离。
- 最后 `rng.shuffle(preference_data)`。

### gt_pairs（evaluate_code.py::main 偏好段）

- k 次重复扁平列表 reshape(k, -1)；任务 i 的第 j 次重复下标 = `j*num_tasks + i`。
- `best_repeat = argmax(arr[:, i])`；对每个 `j != best_repeat 且 arr[j,i] < arr[best_repeat,i]` 出一对 {chosen_index, rejected_index}——**严格 <**（== max 的非 argmax 不出对）。
- 文本组装（本题返回下标，文本账在此钉口径）：GT 变体 chosen/rejected = GT 前缀 + `dialogue_history[-1]` 的 input+output（**最后一个 turn**）；非 GT 变体无前缀（官方代码里 chosen 变量被赋值两次，取同一值——无行为差异）。

## 沙箱与测试

- /tmp/sweet_check.py：官方逻辑直译（含 extract_indices 的 argsort 位置语义、judge 四步提取、pairs 主循环 + Random(42)）与 solution 逐对对拍；judge_score 手工锚（PL：tail=[-0.4,-0.5,-0.6,-20,-21,-22] → mean=-10.75、sum=-64.5；PL_SHORT：tail=[-0.4]）。
- fixture 教训：judge_score 的 fixture 必须数清 [:-4] 截尾与 EOS 位置的相互作用——初版两次把截尾边界算错（tail 混入/少计 x token）；先写「EOS 后恰 1 真实 token + 4 截尾」的短例钉边界，再上长例。
- gt_pairs 的 reshape 方向：扁平 [j0i0, j0i1, j1i0, ...]——测试里先写 reshape 草稿再下断言（初版把 [[1,1,1],[0,0,0]] 当 reshape(3,2) 的行，方向反了）。

## 变异清单（mut792 待做）

易漏点：[:-4] 截尾数、None 跳过、find_last_index 的最后一个 vs 第一个、tail 切片 +1、use_sum 分支、GT 前缀拼接、RESPONSE_TEMPLATE、减基线 vs 不减、argsort 降序、argsort 升序位置语义（前半/后半边界 n//2）、交换条件 >（严格）、get_sft 的 chosen==rejected、no_prompt 拼接顺序、shuffle、gt_pairs 的 argmax / 严格 < / 下标公式。

## 划界

- 789 `rl-tournament-grpo`：judge 成对比较选胜者（win count）；本题 critic 绝对打分（logprob 派生）− 参考基线 + 排序配对。
- 790 `rl-opsd-turn-credit`：OPSD teacher 与 student 同参快照、信念递推乘子；SWEET-RL critic 是独立训练的特权打分模型、产物是 DPO 偏好对。
- to-dpo 系：DPO 损失本身已占；本题只做数据构造侧（critic 打分 → 排序 → 配对）。
