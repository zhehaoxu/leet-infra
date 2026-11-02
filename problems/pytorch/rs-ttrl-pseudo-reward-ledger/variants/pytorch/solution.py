"""TTRL 多数票伪奖励账（arXiv 2504.16084 + PRIME-RL/TTRL main）。

TTRL 在**测试时**用模型自己的多数票当标签：同一道题采 `n` 条，抽出各自的
`\\boxed{}` 答案做投票，得票最多者成为这道题的**伪标签**，然后拿它当
ground truth 反回去给每条样本算奖励，再折算成一组评估指标。

官方实现在 `verl/verl/trainer/ppo/ttrl_utils.py`（`_majority_vote` /
`_batch_majority_vote` / `_batch_compute_ttrl_metrics` / `select_top_k_per_prompt`）
与 `verl/verl/utils/reward_score/ttrl_math/`（`extract_answer` /
`simplify_expression_string` / `grade`）。

`sympy` 化简与 mathd 判等在判题环境不可用，因此本题把这两处**显式化成入参**：

* 化简表 `canon`：`{原始答案: 化简后}`，未命中即原样保留——对应官方
  `simplify_expression_string` 的两级 fallback（parse 失败退 latex2sympy，
  再失败退原串）。每个原始答案只在**进入账本时**归一化一次，之后所有比较
  都拿归一化后的值直接比字符串。
* 判等因此就是「归一化后逐字相同」——官方 `grade` 走 mathd + sympy 等价，
  这里换成字符串相等（note 里明列）。

其余口径逐条对齐官方：

* `extract_answer`：文本里没有 `\\boxed` 子串 → 不过抽（`None`）；有则取
  **最后一次**出现的 `\\boxed`，从该位置起用大括号配平找到闭合，再要求
  候选串以 `\\boxed{` 开头——否则 `None`。配平起点的 `}` 先于 `{` 会让
  计数变负、永远回不到 0，同样是 `None`；`\\boxed{}` 抽出空串 `""`（合法票）。
* 投票：抽出结果里的 `None` 全部丢掉，剩下的逐个归一化；一条都没抽出来 →
  标签是字符串 `"None"`、比例为 `0.0`（**不是** Python 的 `None`）。
* 计票：按首次出现序记 `(答案, 票数)`，`Counter.most_common(1)` 稳定排序，
  并列取**先出现**者。
* `majority_ratio` 的分母是这一块的**样本数**（`n`），不是成功抽出的条数。
* 每条样本的多数票奖励 = 归一化答案是否等于伪标签（没抽出来直接 0）；真值
  奖励 = 归一化答案是否等于真值（真值自身若含 `\\boxed` 先抽一次）。
* 指标：`label_accuracy`（伪标签对不对真值）、`reward_accuracy`（逐条奖励
  与真值奖励**逐样本**相等的比例）、`majority_voting_reward`、
  `ground_truth_reward`、`pass@{n}`（真值奖励之和 `>= 1`），逐题算完再跨题
  取均值，键序取第一题；`majority_ratio` 另外单独平均。
* 降采样 `select_top_k_per_prompt`：每块只留**前 k 条**，官方**不检查
  `k <= n`**，所以 `k > n` 时下标会吃到下一块，最后一块越界直接
  `IndexError`；选完之后整条流水线（含投票与比例分母）都按 `k` 当新的 `n`
  重跑。
"""


def _extract(text):
    """官方 `extract_answer` + `extract_boxed_answer`（大括号配平、取最后一个）。"""
    if "\\boxed" not in text:
        return None
    idx = text.rfind("\\boxed")
    depth = 0
    end = -1
    i = idx
    while i < len(text):
        if text[i] == "{":
            depth += 1
        if text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
        i += 1
    if end < 0:
        return None
    candidate = text[idx:end + 1]
    if candidate[:7] != "\\boxed{" or candidate[-1:] != "}":
        return None
    return candidate[7:-1]


def _canon(value, table):
    """化简表只查一次；未命中原样保留（官方的两级 fallback）。"""
    if table is None:
        return value
    return table.get(value, value)


def _true_value(gt, table):
    """真值自己可能带 `\\boxed`：先抽再化简（官方 `grade` 的第 0 步）。"""
    raw = _extract(gt) if "\\boxed" in gt else gt
    if raw is None:
        return None
    return _canon(raw, table)


def _vote(answers, n):
    """首次出现序计票 + 并列取先者；一条都没有返回 ("None", 0.0)。"""
    tally = []
    seat = {}
    for answer in answers:
        if answer in seat:
            tally[seat[answer]][1] += 1
        else:
            seat[answer] = len(tally)
            tally.append([answer, 1])
    if not tally:
        return "None", 0.0, tally
    best = 0
    for pos in range(1, len(tally)):
        if tally[pos][1] > tally[best][1]:
            best = pos
    return tally[best][0], tally[best][1] / n, tally


def _select(total, n, keep):
    """官方 `select_top_k_per_prompt`：每块前 keep 条，不检查 keep <= n。"""
    selected = []
    for block in range(total // n):
        for step in range(keep):
            index = block * n + step
            if index >= total:
                raise IndexError("list index out of range")
            selected.append(index)
    return selected


def solve(outputs, n, original_gt, canon=None, keep=None):
    outputs = list(outputs)
    original_gt = list(original_gt)

    if n < 1:
        raise ValueError("n must be >= 1")
    total = len(outputs)
    if total % n:
        raise ValueError("outputs length must be divisible by n")

    selected = None
    if keep is not None:
        if keep < 1:
            raise ValueError("keep must be >= 1")
        selected = _select(total, n, keep)
        outputs = [outputs[index] for index in selected]
        n = keep

    if not outputs:
        raise ValueError("empty outputs")

    # 降采样后长度仍是 n(=keep) 的整数倍，这里不再复查整除性。
    num_prompts = len(outputs) // n
    if len(original_gt) != num_prompts:
        raise ValueError("original_gt length must equal the number of prompts")

    prompts = []
    ratios = []
    for block in range(num_prompts):
        chunk = outputs[block * n:(block + 1) * n]
        extracted = [_extract(text) for text in chunk]
        answers = [_canon(item, canon) for item in extracted if item is not None]

        label, ratio, tally = _vote(answers, n)
        ratios.append(ratio)

        rewards = []
        for item in extracted:
            # 没抽出来 → 0；抽出来就只看归一化后的字符串是否等于伪标签。
            rewards.append(0.0 if item is None else
                           1.0 if _canon(item, canon) == label else 0.0)

        truth = _true_value(original_gt[block], canon)
        truth_rewards = []
        for item in extracted:
            if item is None or truth is None:
                truth_rewards.append(0.0)
            elif _canon(item, canon) == truth:
                truth_rewards.append(1.0)
            else:
                truth_rewards.append(0.0)

        hits = 0
        for estimate, actual in zip(rewards, truth_rewards):
            if estimate == actual:
                hits += 1

        metrics = {
            "label_accuracy": 0.0 if truth is None else
                              (1.0 if label == truth else 0.0),
            "reward_accuracy": hits / n,
            "majority_voting_reward": sum(rewards) / n,
            "ground_truth_reward": sum(truth_rewards) / n,
            "pass@%d" % (n,): 1.0 if sum(truth_rewards) >= 1 else 0.0,
        }

        prompts.append({
            "extracted": extracted,
            "answers": answers,
            "tally": tally,
            "label": label,
            "ratio": ratio,
            "rewards": rewards,
            "truth_rewards": truth_rewards,
            "metrics": metrics,
        })

    # 跨题平均：键序取第一题的（官方 `ttrl_metrics[0]` 的键）。
    first = prompts[0]["metrics"]
    metrics = {}
    for key in first:
        metrics[key] = sum(p["metrics"][key] for p in prompts) / len(prompts)
    metrics["majority_ratio"] = sum(ratios) / len(ratios)

    return {
        "n": n,
        "num_prompts": num_prompts,
        "prompts": prompts,
        "metrics": metrics,
        "selected": selected,
    }
