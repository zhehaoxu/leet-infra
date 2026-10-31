# TrainableTokens 定点替换：index_copy 语义与不相交适配器

## 题目描述

给 LLM 加新词（新语言、新领域 token、新 special token）通常要扩词表再全量微调 embedding——太贵。TrainableTokens（huggingface/peft `tuners/trainable_tokens`）走另一条路：**词表不动，只挑出若干个 token 行单独训练**。每个 adapter 拿着一份 `token_indices` 和一块同行数的 delta 矩阵，应用方式是 PyTorch 的 `index_copy`——第 `k` 行 delta **替换**（replace）权重矩阵的第 `token_indices[k]` 行，不是加上去。

「替换而非加法」是上游刻意的设计：delta 在初始化时直接从基座权重的对应行拷贝（`originals`），训练中 delta 就是该行的当前值。若走「加法」，模型重建、embedding 随机重初始化后，行为未定义；替换语义下永远有定义。上游还加了一条硬约束：**任何两个 adapter 的 token indices 不得相交**——重叠合并的行为未定义，直接抛错。本题给定 embedding 权重与一组 adapter，复现这份定点替换：交出合并后的权重与每个 adapter 命中的原始行（供 unmerge 恢复）。

## 任务

实现 `solve(weight, adapters)`：

- `weight`：`[vocab, dim]` 的二维浮点 tensor（Embedding 权重）；
- `adapters`：非空 `list`，每项是含且仅含三个键的 dict：
  - `"name"`：非空字符串，**不得重复**；
  - `"token_indices"`：非空 `list[int]`，每个落在 `[0, vocab)`，**单个 adapter 内不得重复**；
  - `"delta"`：`[len(indices), dim]` 的数值（嵌套 `list` 或二维 tensor），第 `k` 行是新值。

返回 `dict`，键为 `merged_weight`、`originals`（`name` → 该 adapter 命中的原始行，从**最初** `weight` 抓取，形状 `[len(indices), dim]`）。

## 精确规则

1. **替换语义**：`merged` 从 `weight` 复制开始，对每个 adapter **依序**做 `merged = merged.index_copy(dim=0, index=indices, source=delta)`。第 `k` 行 delta 直接成为第 `indices[k]` 行的新值——原值不影响结果（对比 LoRA/SHiRA 的加性）。
2. **不相交检查**：任何两个 adapter 的 `token_indices` 集合有交集 → `ValueError`（上游 `_check_overlapping_tokens`：重叠合并未定义）。adapter 名字重复也抛错。
3. **originals**：对每个 adapter，取 `weight[indices]`（合并**前**的原始行），供 unmerge 恢复用——若错从合并后权重抓取，拿到的会是 delta 自己。
4. delta 行数必须等于 `len(token_indices)`、行宽必须等于 `dim`。

## 示例

```text
W = [[1,1], [2,2], [3,3], [4,4]]（vocab=4, dim=2）
adapter a：indices=[1]，delta=[[9,9]]
adapter b：indices=[3, 0]，delta=[[8,8], [7,7]]

merged = [[7,7], [9,9], [3,3], [8,8]]   # 行 1←a，行 3、0←b，行 2 不动
originals = {a: [[2,2]], b: [[4,4], [1,1]]}
```

语义锚点：`W=[[100,200],[300,400]]`、indices=[0]、delta=[[1,2]] → merged `[[1,2],[300,400]]`——delta 完全决定新行，加法会得 `[[101,202],...]`。

## 约束

- 不修改输入；非法输入（weight 非二维/非有限、adapters 空/非 dict/缺键/名字空或重复、indices 空/非 int/bool/越界/adapter 内重复、delta 行数或行宽不符/非有限、两 adapter 重叠）抛 `ValueError`。
