# TTRL 多数票伪奖励账

TTRL（arXiv 2504.16084）在**测试时**用模型自己的多数票当标签：同一道题采 `n` 条，
抽出各自的 `\boxed{}` 答案投票，得票最多者成为这道题的**伪标签**，再拿它当
ground truth 反回去给每条采样算奖励，最后折成一组评估指标。

本题只保留这条**确定性流水线**（抽取 → 化简 → 投票 → 奖励 → 指标 → 降采样）：
LLM 生成、`sympy` 化简、mathd 判等全部显式化成入参，对应官方
`PRIME-RL/TTRL` 的 `verl/verl/trainer/ppo/ttrl_utils.py` 与
`verl/verl/utils/reward_score/ttrl_math/`。

实现 `solve(outputs, n, original_gt, canon=None, keep=None)`。

## 输入

- `outputs`：扁平字符串列表，长度必须是 `n` 的整数倍；第 `i` 块
  （下标 `[i*n, (i+1)*n)`）是第 `i` 题的 `n` 条采样。
- `n`：每题采样条数，必须 `>= 1`。
- `original_gt`：每题一个真值字符串，条数必须等于题数。
- `canon`：化简表 `{原始答案: 化简后}`，`None` 表示恒等；查不到就原样保留。
- `keep`：可选的降采样条数，见后文。

## 答案抽取

每一条采样先抽答案：

- 文本里没有子串 `\boxed` → 抽不到（`None`）。`\fbox` **不算**。
- 有则从**最后一次**出现的 `\boxed` 开始扫大括号：`{` 加一、`}` 减一，
  **第一次**减到 0 的那个 `}` 是右边界；扫到末尾都没归零 → `None`。
- 截出的候选串必须以 `\boxed{` 开头，否则 `None`——所以 `\boxed` 后面直接跟空格、
  或者配平到的花括号在更后面的情况都抽不到，前面的合法框也不救场。
- 通过则取 `\boxed{` 与末尾 `}` 之间的内容：嵌套花括号原样保留
  （`\boxed{\frac{1}{2}}` → `\frac{1}{2}`），`\boxed{}` 得到**空串**（合法答案）。
- 抽取结果里只有 `None` 表示失败，空串是一条实实在在的票。

## 伪标签与票仓

- 抽不到的采样直接丢掉，剩下的逐个过化简表——每个原始值**只查一次**，之后所有
  比较都用化简后的值直接比字符串。
- 一条都不剩：伪标签是**字符串** `"None"`、比例为 `0.0`、票仓为空
  （它照样参与后面所有判分，所以真值恰好是 `"None"` 时 `label_accuracy` 会给 1.0）。
- 否则按**首次出现序**记 `[答案, 票数]`，票数最多者当选，**并列取先出现者**。
- `ratio` = 当选票数 / **`n`**：分母是这一块的样本数，不是成功抽出的条数。

## 奖励

- 伪标签奖励：抽不到的采样记 0；抽到的看「化简后答案 == 伪标签」。
- 真值奖励：真值自身含 `\boxed` 时先抽一次再化简；抽出来的采样看
  「化简后答案 == 化简后真值」。
- 真值抽不出来（比如 `\boxed{5` 少了右括号）时，这题的真值奖励全是 0，
  连空答案也不给判中。
- 伪标签已经是化简后的值，**不再过表**：链式表（`a->b, b->c`）不会连锁。

## 指标

- 每题五项：
  - `label_accuracy`：伪标签是否等于化简后的真值（真值抽不出来为 0.0）；
  - `reward_accuracy`：逐条伪标签奖励与真值奖励**逐样本**相等的比例；
  - `majority_voting_reward`：伪标签奖励均值；
  - `ground_truth_reward`：真值奖励均值；
  - `pass@{n}`：真值奖励之和 `>= 1` 记 1.0，否则 0.0。
- 跨题取算术均值，**键序取第一题的**；`pass@k` 里的 `k` 是这一批实际生效的每题条数。
- 最后追加一项 `majority_ratio` = 各题比例（`ratio`）的均值，放在末尾。

## 降采样

- 给 `keep` 时先取下标：`[i*n + t for i in 0..题数 for t in 0..keep)`，
  `selected` 就是这个下标列表。
- 官方**不检查** `keep <= n`：`keep > n` 时下标吃到下一块，最后一块越界 → `IndexError`。
- 选完之后整条流水线把 `keep` 当成新的 `n` 重跑：票仓、比例分母、`pass@k` 的 `k`
  全都换成 `keep`。
- `keep` 缺省时 `selected` 是 `None`，`n` 保持原值。

## 输出

```
{
  "n": 4,
  "num_prompts": 1,
  "prompts": [{"extracted": ["5", "5", "6", null],
               "answers": ["5", "5", "6"],
               "tally": [["5", 2], ["6", 1]],
               "label": "5",
               "ratio": 0.5,
               "rewards": [1.0, 1.0, 0.0, 0.0],
               "truth_rewards": [1.0, 1.0, 0.0, 0.0],
               "metrics": {"label_accuracy": 1.0, "reward_accuracy": 1.0,
                           "majority_voting_reward": 0.5,
                           "ground_truth_reward": 0.5, "pass@4": 1.0}}],
  "metrics": {"label_accuracy": 1.0, "reward_accuracy": 1.0,
              "majority_voting_reward": 0.5, "ground_truth_reward": 0.5,
              "pass@4": 1.0, "majority_ratio": 0.5},
  "selected": null
}
```

- `extracted` 逐条给原始抽取结果（失败是 `None`），`answers` 是丢掉 `None` 并化简后的票。
- `tally` 是 `[答案, 票数]` 的列表，按首次出现序。
- `selected` 只有给了 `keep` 才是下标列表。

## 样例

```
outputs = ["so \\boxed{5}", "so \\boxed{5}", "so \\boxed{6}", "no box"]
led = solve(outputs, 4, ["5"])
led["prompts"][0]["label"]          # "5"
led["prompts"][0]["ratio"]          # 0.5
led["prompts"][0]["rewards"]        # [1.0, 1.0, 0.0, 0.0]
led["metrics"]["reward_accuracy"]   # 1.0
led["metrics"]["majority_ratio"]    # 0.5
```

## 约束

- 纯函数：不联网、不读写文件、不修改入参；同样的输入必须给同样的账本。
- 返回的浮点值按普通除法算（`和 / 条数`），比较时按 `1e-12` 容差。
- `n < 1`、`outputs` 长度不是 `n` 的整数倍（降采样前就要查）、`keep < 1`、
  `outputs` 为空、`original_gt` 条数不等于题数 → `ValueError`。
- `keep > n` 导致的越界 → `IndexError`。
