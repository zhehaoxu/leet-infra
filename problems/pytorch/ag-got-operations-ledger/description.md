# Graph of Thoughts 操作账（ag-got-operations-ledger）

**难度**：Medium ｜ **合集**：agentic-systems ｜ **order**：723

把 Graph of Thoughts（GoT，arXiv 2308.09687）官方框架 [spcl/graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) 的 `operations.py` / `thought.py` 在**线性操作链**上无状态化：Thought 的 setter 标志位账、九种操作（score / validate_and_improve / generate / improve / aggregate / keep_best_n / keep_valid / ground_truth / selector）的确定性调度语义逐一记账。LLM 调用全部显式化为入参（scores/valids/updates/parsed/results 序列），prompter/parser/lm 不建模。

## 你要实现的

`solve(ops, base_state=None)`：按序执行操作描述符列表，返回每步账面与最终 thoughts。

## 账点（全部来自官方源码逐行核对）

1. **Thought setter 标志位**：`score` setter 置 `scored=True`、`valid` setter 置 `validated=True`、`solved` setter 置 `compared_to_ground_truth=True`（结果为 False 也置）。缺省 `score=0.0`、其余布尔 False。
2. **两种 thought 产地**：`Score`/`Selector`/`KeepBestN`/`KeepValid`/`GroundTruth`/`ValidateAndImprove` 的初始 attempt 走 `from_thought`（复制全部标志）；`Generate`/`Improve`/`Aggregate` 与 ValidateAndImprove 的**改进产物**走 `Thought({**state, **update})`——**全新缺省 thought，score 复位 0.0、标志全复位**。
3. **Score combined 的 zip 截断**：combined 模式把 scores 与 thoughts zip——scores 更短时多出的 thought **被静默丢弃**；individual 模式逐 thought 赋分。
4. **ValidateAndImprove 的尝试账**：每轮验证消费一个布尔（即使无效也先记账再判停）；`improve=True` 且恒定无效时验证恰 `num_tries+1` 次、改进恰 `num_tries` 次（break 条件是 `current_try >= num_tries`，在改进**之后**计数）。attempt 链末位为最终 thought。
5. **Generate 的空转与合并**：有前驱但前驱零 thoughts → 直接跳过（级联跳过）；新 state 以 `{**父state, **ns}` 合并（父态保留、新键覆盖）；产物超出 `p*r*父数` 只是官方日志警告，**不截断**。
6. **Aggregate 的 score 排序合并**：base_state 按前驱 thoughts **score 升序**（stable，平分保持原序）逐个 `{**base, **state}` 合并——**score 高的后合并、覆盖同名键**（官方注释 "applied in order of score"）；未打分 thought 用缺省 0.0。parsed 为 dict 时包装成单元素列表。
7. **KeepBestN 的两套口径**：全部前驱必须已打分（否则 ValueError）；按 score 排序（stable，平分保原下标序）取前 n。排序失败（如已打分但 score=None）走官方 fallback：**只保留 `isinstance(score, float)` 的 thought**——int score 也会被丢弃。`n<=0` 构造期 ValueError。
8. **KeepValid 的放行 quirk**：保留条件是 `not validated or valid`——**未验证的 thought 直接放行**，只有"已验证且无效"才被剔除。
9. **GroundTruth 的异常吞噬**：求值抛异常 → `solved=False`（不区分异常类型）。
10. **域约束**：首个操作（无前驱）调用 score/validate_and_improve/improve/aggregate/keep_best_n/keep_valid/ground_truth 均 ValueError（官方 assert 转 ValueError）。

## 输入约定

- 每个操作描述符见 starter 注释；`scores`/`valids`/`updates`/`parsed`/`results` 序列按消费序给出，长度保证覆盖用例路径（除 combined score 的截断用例刻意给短）。
- 判题环境无 LLM：全部生成/验证/评分结果已物化为入参。

## 样例

```python
ops = [
    {"op": "generate", "parsed": [[{"v": "a"}, {"v": "b"}]]},
    {"op": "score", "combined": False, "scores": [1.0, 9.0]},
    {"op": "aggregate", "parsed": {"z": 0}},
]
led = solve(ops)
# steps[2]["base_state"] == {"v": "b"}   # 9.0 后合并覆盖 1.0
# steps[2]["thoughts"][0]["state"] == {"v": "b", "z": 0}
# steps[2]["thoughts"][0]["score"] == 0.0  # aggregate 产物全新 thought
```

## 计分

13 个检查点：zip 截断、setter 标志、num_tries+1 尝试账、generate 跳过/合并、aggregate 排序覆盖、KeepBestN 平分与 fallback（含 int score 丢弃）、KeepValid 放行、GroundTruth 异常、selector 标志复制、improve 复位、域约束，以及 100 组随机操作链对拍。
