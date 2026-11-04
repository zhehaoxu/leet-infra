"""SWEET-RL 的 turn-level critic 账。

回源: facebookresearch/sweet_rl scripts/（rank_best_of_n.py +
generate_random_pairs_from_ranks.py + sample_best_of_n.py +
evaluate_code.py），论文 arXiv:2503.15478。训练在配套 fork
YifeiZhou02/collab_openrlhf（multi-turn DPO + length normalization）。

SWEET-RL 的核心主张：多轮 RL 的 credit assignment 用一个**带
training-time 信息**的 turn-level critic——critic 的 prompt 里注入
ground truth（"In light that the final answer is: ..."），给每个 turn
的候选输出打分；policy 再用 critic 排序生成的 step-level 偏好对做
（length-normalized）DPO。模型调用全部入参化：logprob 列表、token
列表、judge 分数、参考基线分数均为显式入参，只保留脚本里的确定性账。

五入口复刻:
- flatten: sample_best_of_n 的 dialogue_history 展平（每 turn 一条、
  input_with_ground_truth 注入前缀）
- judge_prompt: rank_best_of_n 的 prompt 构造（ground truth 前缀 +
  RESPONSE_TEMPLATE 尾巴）
- judge_score: prompt_logprobs 提取账——去掉尾部 4 个空响应 token、
  跳过 None、取**最后一个 eos 之后**的 logprob 序列做 sum 或 mean
  （use_sum = 长度归一开关）
- rank: judge 分数减参考基线（同形状 reshape(-1, n)）后 argsort 降序
  rerank
- pairs: generate_random_pairs_from_ranks 账——升序 argsort 后 chosen
  从前半随机挑、rejected 从后半随机挑、按实际分数交换保证 chosen 更高
  （bestworst 变体直接取首尾）；no_prompt 时 input 拼进 response 侧；
  random.seed(42) 固定抽签与 shuffle
- gt_pairs: evaluate_code 的 ground-truth 偏好账——k 次重复按任务
  argmax 选 best、其余 reward < max 的轨迹作 rejected；chosen/rejected
  文本带 ground truth 前缀 + 最后一个 turn 的 input+output
"""

import random

import numpy as np

RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"
GT_PREFIX = "In light that the final answer is: {ground_truth}."


def find_last_index(lst, element):
    """最后一个匹配元素的下标；不存在返回 -1（官方实现直译）。"""

    try:
        return len(lst) - 1 - lst[::-1].index(element)
    except ValueError:
        return -1


def flatten_turns(dialogue_history, ground_truth):
    """sample_best_of_n 的展平账：每个带 input 的 turn 一条记录。

    input_with_ground_truth = "In light that the final answer is: {gt}."
    + input（critic 的 training-time 信息注入点）；messages 截到当前
    turn（[:i+1]）。"""

    records = []
    for i, turn in enumerate(dialogue_history):
        if "input" in turn:
            records.append(
                {
                    "input": turn["input"],
                    "input_with_ground_truth": GT_PREFIX.format(
                        ground_truth=ground_truth)
                    + turn["input"],
                    "older_output": turn["output"],
                    "messages": list(dialogue_history[: i + 1]),
                }
            )
    return records


def judge_prompt(turn_input, output, ground_truth=None,
                 response_template=RESPONSE_TEMPLATE):
    """rank_best_of_n 的 prompt 构造：ground truth 前缀（training-time
    info）+ 交互文本 + RESPONSE_TEMPLATE 尾巴；no_use_ground_truth 时
    不注入前缀。"""

    if ground_truth is None:
        return turn_input + output + response_template
    return (
        GT_PREFIX.format(ground_truth=ground_truth)
        + turn_input
        + output
        + response_template
    )


def extract_judge_score(prompt_logprobs, eos_token, use_sum=False):
    """rank_best_of_n 的 judge 分数提取账。

    prompt_logprobs 先去掉**尾部 4 个**（官方注释：the last four digits
    are another empty response），逐位跳过 None（logprob 与 decoded
    token 同步收集）；token 序列里找**最后一个 eos**，取其后所有
    logprob：use_sum 时求和（sum 变体），否则除以个数（mean 变体 =
    length normalization）。每项 prompt_logprob 是 {token_id:
    (logprob, decoded_token)} 形式的 dict（官方 Logprob 对象的直译）。"""

    logprobs = []
    all_tokens = []
    for prompt_logprob in prompt_logprobs[:-4]:
        if prompt_logprob is not None:
            for value in prompt_logprob.values():
                logprob, decoded_token = value
                logprobs.append(logprob)
                all_tokens.append(decoded_token)
    last_occurence = find_last_index(all_tokens, eos_token)
    tail = logprobs[last_occurence + 1:]
    if use_sum:
        return sum(tail)
    return sum(tail) / len(tail)


def rank_against_reference(judge_scores, reference_scores, best_of_n):
    """judge 分数减同形状参考基线后 argsort 降序，逐组返回 rerank 下标。

    官方：judge_results = judge - reference（reshape(-1, best_of_n)），
    再 torch.argsort(descending=True, dim=1) 重排 best-of-n 候选。"""

    judge = np.asarray(judge_scores, dtype=float).reshape(-1, best_of_n)
    reference = np.asarray(reference_scores, dtype=float).reshape(
        -1, best_of_n)
    adjusted = judge - reference
    return [list(map(int, row))
            for row in np.argsort(-adjusted, axis=1)]


def extract_pair_indices(judge_results, num_outputs, rng,
                         bestworst=False):
    """generate_random_pairs_from_ranks 的抽签账。

    np.argsort **升序**后：chosen 候选从前半 [0, n/2) 随机挑、rejected
    候选从后半 [n/2, n) 随机挑（官方 random.seed(42) 固定），再按实际
    分数交换保证 chosen 分数更高——即 pair 永远是「前半挑的 vs 后半挑
    的」中分高者为 chosen（避免永远取绝对最优的过拟合）。bestworst
    变体直接取升序首（最低分）尾（最高分）。"""

    argsorted_indices = np.argsort(judge_results)
    if bestworst:
        return [int(argsorted_indices[0]), int(argsorted_indices[-1])]
    chosen_index = rng.choice(list(range(num_outputs // 2)))
    rejected_index = rng.choice(list(range(num_outputs // 2, num_outputs)))
    return [int(argsorted_indices[chosen_index]),
            int(argsorted_indices[rejected_index])]


def build_preference_pairs(data, num_pairs=8, seed=42, no_prompt=False,
                           get_sft=False, eos_token="<|eot_id|>",
                           bestworst=False):
    """generate_random_pairs_from_ranks 的主账（模型无关部分）。

    data 每项 {input, additional_outputs, judge_result}。num_outputs =
    len(additional_outputs)；每题抽 num_pairs 对；get_sft 时官方直接
    random.sample 两个下标并**把 chosen 与 rejected 都设为第一个**
    （官方原样的 SFT 姿势）；否则走 extract_pair_indices + 分数交换。
    文本组装：no_prompt 时 chosen/rejected = input + output + eos
    （上下文进 response 侧），否则 output + eos 与 prompt 分离；
    get_sft 恒带 prompt。最后 random.shuffle 打乱。返回逐对 dict 列表
    （含 chosen_index/rejected_index 诊断字段）。"""

    rng = random.Random(seed)
    preference_data = []
    for d in data:
        num_outputs = len(d["additional_outputs"])
        for _ in range(num_pairs):
            if get_sft:
                indices = rng.sample(range(num_outputs), 2)
                chosen_index = indices[0]
                rejected_index = indices[0]
            else:
                indices = extract_pair_indices(
                    d["judge_result"], num_outputs, rng, bestworst)
                if d["judge_result"][indices[0]] > d["judge_result"][indices[1]]:
                    chosen_index = indices[0]
                    rejected_index = indices[1]
                else:
                    chosen_index = indices[1]
                    rejected_index = indices[0]
            preference_data.append(
                {
                    "chosen": {
                        "input": d["input"],
                        "output": d["additional_outputs"][chosen_index],
                    },
                    "rejected": {
                        "input": d["input"],
                        "output": d["additional_outputs"][rejected_index],
                    },
                    "chosen_index": chosen_index,
                    "rejected_index": rejected_index,
                }
            )

    rng.shuffle(preference_data)

    prepared = []
    for d in preference_data:
        if get_sft:
            prepared.append(
                {
                    "chosen": d["chosen"]["output"] + eos_token,
                    "rejected": d["rejected"]["output"] + eos_token,
                    "prompt": d["chosen"]["input"],
                }
            )
        elif no_prompt:
            prepared.append(
                {
                    "chosen": d["chosen"]["input"]
                    + d["chosen"]["output"]
                    + eos_token,
                    "rejected": d["rejected"]["input"]
                    + d["rejected"]["output"]
                    + eos_token,
                }
            )
        else:
            prepared.append(
                {
                    "chosen": d["chosen"]["output"] + eos_token,
                    "rejected": d["rejected"]["output"] + eos_token,
                    "prompt": d["chosen"]["input"],
                }
            )
    return prepared


def ground_truth_preferences(rewards, k, add_prefix=True):
    """evaluate_code 的 ground-truth 偏好账（轨迹级 RM 训练数据）。

    rewards 是 k 次重复的扁平列表（任务 i 的第 j 次重复在下标
    j*num_tasks+i——reshape(k, -1)）。每任务取 argmax 重复为 best；对
    每个 j != argmax 且 reward < max 的重复出一对 {chosen, rejected}
    下标。文本组装由调用方做，这里返回对的下标与是否前缀注入的
    chosen/rejected **turn 文本**（dialogue_history 最后一项的
    input+output，ground truth 前缀按 add_prefix 开关）。"""

    arr = np.asarray(rewards, dtype=float).reshape(k, -1)
    num_tasks = arr.shape[1]
    pairs = []
    for i in range(num_tasks):
        best_repeat = int(np.argmax(arr[:, i]))
        best_index = num_tasks * best_repeat + i
        for j in range(k):
            if j != best_repeat and arr[j, i] < arr[best_repeat, i]:
                pairs.append(
                    {
                        "chosen_index": best_index,
                        "rejected_index": num_tasks * j + i,
                    }
                )
    return pairs


def solve(mode=None, **kw):
    if mode == "flatten":
        return flatten_turns(kw["dialogue_history"], kw["ground_truth"])
    if mode == "judge_prompt":
        return judge_prompt(
            kw["turn_input"], kw["output"],
            ground_truth=kw.get("ground_truth"),
            response_template=kw.get("response_template",
                                     RESPONSE_TEMPLATE))
    if mode == "judge_score":
        return extract_judge_score(
            kw["prompt_logprobs"], kw["eos_token"],
            use_sum=kw.get("use_sum", False))
    if mode == "rank":
        return rank_against_reference(
            kw["judge_scores"], kw["reference_scores"], kw["best_of_n"])
    if mode == "pairs":
        return build_preference_pairs(
            kw["data"], num_pairs=kw.get("num_pairs", 8),
            seed=kw.get("seed", 42), no_prompt=kw.get("no_prompt", False),
            get_sft=kw.get("get_sft", False),
            eos_token=kw.get("eos_token", "<|eot_id|>"),
            bestworst=kw.get("bestworst", False))
    if mode == "gt_pairs":
        return ground_truth_preferences(
            kw["rewards"], kw["k"], add_prefix=kw.get("add_prefix", True))
    raise ValueError("unknown mode: %s" % mode)
