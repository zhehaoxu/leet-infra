# notes — ag-tot-bfs-ledger (order 720)

## 回源口径

- 官方仓 princeton-nlp/tree-of-thought-llm，**源码在 `src/tot/`**（master 分支；README 写的 `tot/methods/bfs.py` 路径已过时，raw 老路径 404）。
- `solve` 主循环：`ys = ['']` 起步，每步 生成 → 打分 → 选择 → `infos.append({step,x,ys,new_ys,values,select_new_ys})` → `ys = select_new_ys`。本题照抄该结构，评估走 `value` 分支、选择走 `greedy` 分支。
- game24 `value_outputs_unwrap`：`if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower(): return 0`；`value_names = [_.split('\n')[-1] for _ in value_outputs]`；`value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}`（**官方原注 "TODO: ad hoc"**）；`value = sum(value * value_names.count(name) ...)`——**不除以 len(value_outputs)**。
- `get_values` 的 local_value_cache：`if y in local_value_cache: value = 0`——重复候选硬置 0 **而非查缓存值**，且只在首次计算时写入缓存。

## 关键账点

1. **重复候选置 0**：同一父节点提出相同 proposal（如输出 "x\nx"）→ 第二个候选值 0，其评估文本被无视。变异 `value = 0 → value = local_value_cache[y]` 与 `if y in cache → if False` 均由该锚点抓出。
2. **权重表不归一化**：2 条 sure = 40.0；变异加 `/ len(eval_outputs)` 会被"空评估列表 → 0"退化用例与 40 锚点双抓。
3. **评估文本只看末行**：`sure\nbecause of x` 的末行不是权重表键 → 计 0；大小写敏感（`SURE` 不匹配）。
4. **4 行无 answer 惩罚**：strip 后行数（propose 子候选恒带尾 `\n`——`"a\nb\nc\n".strip()` 是 3 行不触发；sample 样本自带 `\n` 可一次成 4/5 行）。`== 4` 与 `>= 4` 需 5 行锚点区分；`'answer' in y.lower()` 的大小写不敏感豁免与原样匹配变异由 `Answer: x` 锚点区分。
5. **拼接差异**：propose `y + p + '\n'` vs sample `y + s`——变异（去 `\n` / 加 `\n`）由 new_ys 精确断言与跨步 4 行行为共同抓。
6. **greedy 稳定排序**：`sorted(ids, key=value, reverse=True)` 平分保持原下标序；变异 `key=(value, id)` 翻转平分序被抓。
7. **propose 每步只加一行**：4 行惩罚需 3 步 propose 链累积或 sample 一次成型——手算锚点两种都放。

## 官方不建模（记档）

- `task.value_cache`（跨步、跨候选的 prompt 级缓存）：语义依赖真实 prompt 构造，不入题；重复判定的 local cache（每步重建）已入题。
- `method_evaluate='vote'`（`get_votes`）与 `method_select='sample'`（`np.random.choice` 默认**有放回** + 全局 RNG）：随机性红线，不入题。
- 官方对未知 `method_generate` **无 else 分支** → `NameError: new_ys`（实际是引用未定义变量）；本题改 `ValueError`，差异记档。
- game24 的 `propose_prompt_wrap`/`get_current_numbers`（`left: ` 解析）属 prompt 构造侧，评估文本已入参，不入题。

## 出题过程教训

- **oracle 浮点求和序必须与官方一致**：官方 `sum()` 按权重表序折叠（imp→likely→sure），oracle 初版按评估文本顺序累加，`40.001` vs `40.001000000000005` 的 1e-15 差把两个"等值"候选的排序翻过来（random 第 129 组抓出）。oracle 改计数法同序折叠后消失。
- **evals[i] 是逐子候选一条**（每条是该候选的评估文本列表），不是逐评估文本——手写锚点时两次搭错形状。
- **propose 一步只加一行**：想构造 4 行 y 只能多步链或改用 sample——锚点设计时先算清每个候选的实际行数。
- 等价变异 3 条（infos 里的 `list(...)` 冗余拷贝：`ys/new_ys/select_new_ys` 均不被就地修改）实测 SURVIVED，记档防后续误判。
- 变异 18 条全 CAUGHT。
