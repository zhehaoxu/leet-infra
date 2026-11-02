# ToT+BFS 搜索过程账

## 题意

Tree of Thoughts（Yao et al., 2305.10601）把 CoT 的单链生成换成树搜索：每一步从当前状态展开多个候选思考，用评估器打分，只保留最好的 b 个继续。官方实现（princeton-nlp/tree-of-thought-llm）的 `src/tot/methods/bfs.py::solve` 就是这套 ToT + BFS 的主循环，配 Game of 24 的价值头（`src/tot/tasks/game24.py`）。

模型生成与评估都是不可判定的 LLM 调用，本题把它们**全部作显式入参**，只考搜索循环本身的账：

- **展开**：`propose` 模式下每条模型输出按 `\n` 切成 proposals，子候选 = `y + p + "\n"`（官方拼 **尾部换行**）；`sample` 模式下子候选 = `y + s`（**无分隔符**）。候选按父节点优先序展平（`itertools.chain` 语义），提供的父槽位多于实际父节点数时多余槽位被无视。
- **打分**（game24 价值头）：候选的评估文本取**每条的最后一行**精确匹配权重表 `{impossible: 0.001, likely: 1, sure: 20}`（官方原注 ad hoc），求和**不除以样本数**（两条 sure = 40 而非 20）；候选恰 4 行且不含 `answer`（大小写不敏感）直接 0；**同一步内重复出现的候选直接 0**（官方 `get_values` 的 local cache quirk——第二次出现不是查缓存值，是硬置 0，其评估文本被无视）。
- **选择**：greedy 稳定降序取前 b，**平分保持原下标序**（小 id 优先）。
- **接力**：本轮 `select_new_ys` 即下一轮父节点。

逐步输出账本：`step / x / ys / new_ys / values / select_new_ys`，末轮选择作为 `final`。

## 求解

实现 `solve(x, plan, n_select_sample=5)`。域约束：`n_select_sample` 为正整数；plan 非空且每步含 `method/gen/evals`；未知 method、槽位缺 gen/evals、`evals[i]` 与该父节点的子候选数不匹配 → `ValueError`。

## 样例

```python
solve("1 2 3 4", [
    {"method": "propose", "gen": ["a\nb"],
     "evals": [[["sure", "sure"], ["likely", "impossible"]]]}
], n_select_sample=5)
# new_ys = ["a\n", "b\n"]（propose 拼尾部换行）
# values = [40.0, 1.001]（2×sure 不除以样本数；0.001+1）
```

## 数据范围

- `1 ≤ 步数 ≤ 3`，每步父槽位 ≤ 5，`b ∈ {2,3,5}`；候选/评估文本均为短字符串。
- 200 组随机对拍 + 全锚点精确断言。
