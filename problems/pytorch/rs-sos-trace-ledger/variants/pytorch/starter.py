"""Stream of Search 的轨迹序列化账。

把 Stream of Search（arXiv 2404.03683）在 Countdown 上的「搜索过程即语言」
摊成一本可逐行核对的账：一个纯启发式驱动的深度优先搜索把每一步状态、操作、
前进/回溯**编码**成一段文本轨迹，官方的解析器再把这段轨迹**解码**回一条操作
序列逐步重放，最后按「用掉多少节点 / 满树有多少节点」折成 0~1 的评级。对应
官方 `kanishkg/stream-of-search` 的 `src/countdown_dfs.py::dfs` 与
`src/countdown_utils.py`（`combine_nums` / 两个启发式 / `parse_trajectory` /
`metric_fn`）。

入参：

```
target        int          # 目标数
nums          list[int]    # 初始数字，2~5 个、非负
heuristic     str          # "sum" 或 "mult"
threshold     int|None     # 只保留启发式 <= threshold 的子节点
trace         str|None     # 给了就直接解码这段轨迹，nums/target 不参与
```

你实现的 `solve(target, nums, heuristic="sum", threshold=None, trace=None)`
返回：

```
{
  "trace": str,               # 轨迹全文（每行以 \n 收尾）
  "encoded": bool,            # True = 这次是现场编码的
  "found": bool,              # 轨迹里有没有 Goal Reached
  "states": int,              # "Current State" 出现次数
  "moves": int,               # "Moving to Node" 出现次数
  "explored": int,            # "Exploring Operation" 出现次数
  "node_ids": [str],          # 每个 "Generated Node #<id>" 的 id，按写出序
  "ops": [str],               # 每条探索的操作，按写出序
  "final": {"value": int, "target": int, "equal": bool}|None,
  "verdict": str,             # 解码判词
  "score": int,               # explored + 1
  "max_nodes": int|None,      # 满树节点数（按初始数字个数查表）
  "rating": float,            # 合法时 max(1 - score/max_nodes, 0.0)，否则 0.0
}
```

参数与规则见 `solution.py` 的 docstring 与题面。
"""


def solve(target, nums, heuristic="sum", threshold=None, trace=None):
    raise NotImplementedError
