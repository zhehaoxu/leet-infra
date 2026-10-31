# 参考做法：替换语义、不相交契约与 originals 时机

## 实现

1. 校验顺序：weight 二维有限 → adapters 非空、逐项 dict 且**键集合恰好**是 name/token_indices/delta → name 非空且全局唯一 → indices 非空、逐项 int（拒 bool）、范围 `[0, vocab)`、**adapter 内唯一**（上游 config 允许重复 index 吗不确定，但 `index_copy` 对重复 index 的行为未定义，本题钉死唯一并校验）→ delta（tensor 或嵌套 list）行数 = len(indices)、行宽 = dim、逐项有限。
2. **不相交检查**：两两 `set` 交集非空即抛。上游 `_check_overlapping_tokens` 把 `merged_adapters + adapter_names` 用 `dict.fromkeys` 去重后收集 index 集合——效果等价于「所有 adapter 的 index 集合两两不相交」（同一 adapter 名出现两次先被去重）。名字重复在本题入口即拒。
3. **originals 先抓**：在 index_copy 循环**之前**从原始 weight 取 `weight[idx]`——这是上游 update_layer（merge 前）抓 `trainable_tokens_original` 的时机。错序成 merge 后抓，originals 会等于 delta 自己，是本题第二雷。
4. **依序 index_copy**：`merged.clone()` 后逐 adapter `index_copy(dim=0, ...)`。由于不相交是硬契约，依序应用与顺序无关（任一排列结果相同）——这既是推导性质，也让「反转 adapter 顺序」成为**等价变异**，进不了有效变异集。

## 正确性检验

4 个 `check_*` 覆盖：题面手算示例（两个 adapter、originals 从原始行抓）、替换 vs 加法锚点（W=[100,200] 处 delta=[1,2] 直接换行）、torch `index_copy` 对拍（固定种子随机 8×5、两个 adapter、delta tensor/list 两态、dim=1 窄表、originals 与 `W[idx]` 一致）、重叠两态（完全/部分）与 13 条非法输入。

变异测试（4/4 有效变异全抓 + 2 个等价变异记录）：index_copy 改加法（`merged[index] = merged[index] + delta`，锚点与示例全抓）、不相交检查删除（重叠用例抓）、originals 挪到 merge 后从 merged 抓（示例抓）、adapter 内重复 index 放行（validation 抓）。等价变异两个：① 反转 adapter 应用顺序——不相交契约下顺序无关，是推导性质；② `weight + weight.sum()*0` 形式的 no-op 数值等价写法。

## 范围与来源

上游 huggingface/peft `src/peft/tuners/trainable_tokens/layer.py`：

- `TrainableTokensLayer.update_layer`（L109-160）：`originals = self.weight[self.token_indices[adapter_name]]`（merge 前抓原始行）；`init_weights=True` 时 delta 直接取 originals（恒等启动）。
- `_check_overlapping_tokens`（L162-183）：`dict.fromkeys(merged_adapters + adapter_names)` 收集 index 集合，交集非空抛 `ValueError`（"Only disjunct token indices are currently supported"）。
- `get_merged_weights`（L220-233）：对每个 active adapter 依序 `W = W.index_copy(dim=0, index=index, source=deltas)`——替换语义的出处。
- `merge`（L185-206）/`unmerge`（L208-218）：merge 用 delta 覆盖、unmerge 用 `trainable_tokens_original` 覆盖——originals 的存在意义。
- `forward_adapters`（L235-281）：Embedding 走 `F.embedding`、Linear（LM head 权重绑定）走 `F.linear`；`_get_embed_scale` 的 Gemma 缩放接线不出题。

本题的口径差异：

- **不跑模型**：只考权重合并语义（Embedding 权重口径）；Embedding/Linear 前向、weight tying（`tied_adapter`）、DeepSpeed zero3 的 `_collect_token_weights` 广播不出。
- delta 直接入参（上游是训练中的 Parameter），`init_weights` 的恒等初始化不出。
- adapter 内 index 唯一性：上游未在 layer 层显式校验（`index_copy` 对重复 index 行为未定义），本题钉死唯一、显式抛错——把未定义行为变成确定契约。
- 返回契约显式化：`originals` 按名字分组返回（上游存 `BufferDict`）。
