"""TTRL 的多数票伪奖励账。

把 TTRL（arXiv 2504.16084）的测试时伪标签流水线摊开成一本可逐项核对的账：
同一道题采 `n` 条 → 逐条抽出 `\\boxed{}` 答案 → 化简后多数投票得到伪标签
→ 拿伪标签与真值分别算奖励 → 折成指标。对应官方 `PRIME-RL/TTRL` 的
`ttrl_utils.py`（`_majority_vote` / `_batch_compute_ttrl_metrics` /
`select_top_k_per_prompt`）与 `reward_score/ttrl_math/`。

入参：

```
outputs       list[str]   # 扁平采样结果，长度必须是 n 的整数倍；每 n 条一题
n             int         # 每题采样条数
original_gt   list[str]   # 每题一个真值，长度 = len(outputs) // n
canon         dict|None   # 化简表 {原始答案: 化简后}，未命中原样保留
keep          int|None    # 降采样：每块只留前 n 条里的前 keep 条
```

`canon` 与 `keep` 是官方 `simplify_expression_string`（sympy）与
`select_top_k_per_prompt` 的显式化：化简表查不到就原样返回，降采样只取下标。

你实现的 `solve(outputs, n, original_gt, canon=None, keep=None)` 返回：

```
{
  "n": int,                    # 实际生效的每题条数（降采样后 = keep）
  "num_prompts": int,
  "prompts": [{"extracted": [str|None],   # 逐条抽出的原始答案
               "answers": [str],          # 丢掉 None 并化简后的票
               "tally": [[str, int]],     # 首次出现序的票仓
               "label": str,              # 伪标签
               "ratio": float,            # 得票 / n
               "rewards": [float],        # 逐条多数票奖励
               "truth_rewards": [float],  # 逐条真值奖励
               "metrics": {...}}],
  "metrics": {...},            # 跨题均值 + majority_ratio
  "selected": [int]|None,      # 降采样选中的原始下标
}
```

参数与规则见 `solution.py` 的 docstring 与题面。
"""


def solve(outputs, n, original_gt, canon=None, keep=None):
    raise NotImplementedError
