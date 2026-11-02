# FunSearch 岛屿程序库：聚类注册、温度采样与弱岛重置账

FunSearch 用 LLM 不断改写一个小函数，但真正决定「进化往哪走」的不是模型，而是它背后的
**程序库**：程序按逐测试分数向量聚成簇，取样概率由「簇分 / 温度」的 softmax 给出，温度随
该岛累计程序数线性衰减；一批岛各跑各的，弱的那一半定期清空、把强岛的最优程序克隆过去当
创始人。本题就是把 `google-deepmind/funsearch` 的 `ProgramsDatabase` 搬成纯函数账。

## 接口

```python
def solve(events, num_islands=10, functions_per_prompt=2,
          cluster_sampling_temperature_init=0.1,
          cluster_sampling_temperature_period=30000,
          function_name="f") -> dict
```

`events` 是事件序列，三种：

| `kind` | 字段 | 含义 |
| --- | --- | --- |
| `register` | `program`、`scores`、`island` | 把一个程序登记进某岛；`island=None` 表示种子程序，广播到所有岛 |
| `prompt` | `island`、`clusters`、`programs` | 该岛要取一次提示词；`clusters`/`programs` 是**显式给出的取样结果** |
| `reset` | `founders` | 重置弱半岛；`founders` 显式给出每个被重置岛的创始人岛号 |

`program` 是 `{"id": str, "doc": str, "size": int}`——`size` 是源码长度，簇内取样按它加权。

返回 `clusters`（每岛的簇表，含 `signature`/`score`/`programs`）、`cluster_count`、
`num_programs`、`best_score`、`best_program`、`registrations`、`prompts`、`resets`。
其中 `prompts[i]` 带 `temperature`、`cluster_scores`、`probabilities`、`chosen_clusters`、
`chosen_programs`、`program_probs`、`sorted_clusters`、`versions`、`version_generated`、
`header`。

## 输入输出

```
输入：num_islands=2, functions_per_prompt=2, function_name="f"
     events=[{kind:"register", island:0,
              program:{id:"p0", doc:"returns 0", size:12},
              scores:{"t2": 9.0, "t1": 1.0}},
             {kind:"prompt", island:0, clusters:[0], programs:[0]}]
输出：clusters[0] = [{signature:[1.0], score:1.0, programs:["p0"]}]
     best_score[0] = 1.0
     prompts[0].header = {name:"f_v1", docstring:"Improved version of `f_v0`."}
     prompts[0].version_generated = 2
```

`{"t2": 9.0, "t1": 1.0}` 的簇键是**按测试名升序**的 `[1.0]`，而簇分取的是**最后插入**
那个测试 `t1` 的 `1.0`——不是看着更大的 `9.0`。这条是本题最容易踩的一脚。

## 规则

- **簇键**：`tuple(scores[k] for k in sorted(scores))`——逐测试分数按测试名升序排列。
  两个程序只有**全部分项分数**都相同才算同簇；簇分不同也算不同簇。
- **簇分**：`scores[list(scores)[-1]]`，即**插入顺序最后**那个测试的分数。只在首个成员
  进簇时定下；后来者只往 `programs` 里追加，**不改簇分**，即使它的簇分更高。
- **广播**：`island=None` 时对每个岛各登记一次，`num_programs` 每岛各加一。
- **岛内最优**：只有 `score > 当前最优` 才换人（严格大于，同分留先到者），同时记下
  `best_program` 与该次的逐测试分数（重置时要用）。
- **温度**：`T = t_init * (1 - (num_programs % period) / period)`，其中 `num_programs`
  是**该岛**累计登记数（因广播与重置克隆都会加一）。`n % period == 0` 时回到 `t_init`。
- **簇概率**：`softmax(cluster_scores / T)`，簇顺序即插入顺序。
- **取几个簇**：`min(簇数, functions_per_prompt)`，**有放回**（同一簇可被取两次）。
- **版本编号**：取出的簇按**簇分升序**排，编号 `f_v0 … f_v(n-1)`；`f_v0` 保留原 docstring，
  `f_vi`（i≥1）的 docstring 换成 `Improved version of \`f_v{i-1}\`.`；最后再补一个空壳
  `f_vn`，其 docstring 是 `Improved version of \`f_v{n-1}\`.`。`version_generated = n + 1`。
- **簇内取样概率**：`normalized = (size - min) / (max + 1e-6)`，再取 `softmax(-normalized, 1.0)`。
  注意分母是 `max + 1e-6` 而不是 `max - min`——照抄官方实现，长度越短概率越高但不会塌成 argmin。
- **重置**：按 `(岛内最优, 岛号)` 升序排，前 `num_islands // 2` 个是弱岛（`num_islands=1`
  时一个都不重置）。每个弱岛清空成空岛、最优置 `-inf`，然后克隆创始人岛的 `best_program`，
  于是新岛的簇表只有这一簇、`num_programs=1`、最优分随克隆继承。
- **校验**：`num_islands`/`functions_per_prompt`/`period` 必须是 `>=1` 的整数（`bool` 拒收）；
  `t_init` 必须有限且为正；分数必须有限、非 `bool`；`scores` 不得为空；空岛取提示词、
  簇或程序下标越界、`clusters` 长度不等于 `min(簇数, fpp)`、创始人不在保留岛或手上没程序、
  `founders` 数量不等于 `num_islands // 2`、未知 `kind`——一律 `ValueError`。

## 提示

- 同一条 `scores` 字典，插入顺序不同会给出**不同的簇分**（键序不变、簇分变）。用
  `dict` 的 `list(scores)[-1]` 直接取最后一个键即可，别去 `max` 或 `sorted`。
- 温度是「岛」的属性，不是「库」的属性：广播种子让每个岛都涨一次 `num_programs`，
  重置后新岛的计数从 0 重新开始，这两处都会改变后续的取样概率。
- `-0.0` 与 `0.0` 在 Python 的 dict 里是**同一个键**（`hash` 相同且 `==`），所以
  `{"t": -0.0}` 与 `{"t": 0.0}` 落进同一簇；这不是 bug，是本题的对齐口径。
