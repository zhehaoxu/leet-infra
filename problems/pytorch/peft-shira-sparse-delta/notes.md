# 参考做法：COO 累加散点与加性合并

## 实现

1. 校验顺序：weight 二维有限 → indices 非空、逐项二元组与整数坐标（拒 bool）、范围检查 → **预算** `K <= in*out` → values（list 或一维 tensor，长度 = K，逐项有限）→ scaling 非负有限（拒 bool 与 inf）→ x 至少一维、末维 = in、有限。
2. **散点重建**：`delta[r, c] += v * scaling`。用 float64 累加再转回 weight dtype，与上游 sparse 路径的浮点行为一致。等价写法是 `torch.sparse_coo_tensor(idx, values*scaling, shape).to_dense()`——**不要**先 coalesce 再比较：coalesce 本身就是求和，两种写法同义，但「dict 覆盖」式重建是本题第一雷。
3. **加性合并**：`merged = weight + delta`。丢掉 delta（merged=weight）或前向用基座权重（`x @ weight.T`）都是雷——上游 forward 是把 delta 加进基座后一次 `F.linear`。

## 正确性检验

4 个 `check_*` 覆盖：题面手算示例（重复坐标累加 6、merged、y）、固定种子 20 坐标对拍（oracle 直接用 `torch.sparse_coo_tensor(...).to_dense()` 重建——上游同款构造，含大量自然重复坐标）、scaling=0.5 等价 values 减半、values 用一维 tensor 传入、一维 x、W=0 加性锚点（`y=[[4,0]]`）、预算恰好 `K=in*out` 合法、14 条非法输入。

oracle 的关键设计：不重写散点循环，而是直接调 `torch.sparse_coo_tensor` 的 `to_dense`——这是上游 `get_delta_weight` 的原语，天然携带「重复坐标累加」语义，与本实现的 Python 循环互为印证。torch 端会发 sparse invariant 的 UserWarning（未显式 opt-in 的常规提示），不影响结果。

变异测试（6/6 全抓）：重复坐标改覆盖、丢失 scaling、merged 丢 delta、y 用基座权重、预算检查删除、scaling 负数放行。预算变异由 `check_validation` 的 `K=7 > 6` 用例抓住。

## 范围与来源

上游 huggingface/peft `src/peft/tuners/shira/layer.py`：

- `ShiraLayer.update_layer`（L78-126）：`num_shira_weight = r*(in+out)`，`> in*out` 抛错（预算检查）；mask → `torch.where(mask == 1.0)` → `torch.cat` 成 2×K 的 int 索引；`shira_weight` 是长度 K 的**稠密一维参数**（存 vector、散点时才用 sparse tensor，规避 PyTorch 稀疏参数的 autograd 缺陷）。
- `Linear.get_delta_weight`（L210-226）：`torch.sparse_coo_tensor(indices, weight*scaling, (out, in))`——scaling 乘在值上、delta 是 COO 稀疏张量。
- `Linear.forward`（L228-259）：`new_weight = get_base_weight().clone()`，`new_weight += delta`，`F.linear(x, new_weight, bias)`——合并后权重的前向。
- `Linear.merge`（L158-196）：`orig_weight += delta_weight`——加性合并。

本题的口径差异：

- **坐标直接入参**：省去 `r` 与 mask 的间接层（`K` 即坐标数，预算检查保留）；上游 mask 路径的行优先序（torch.where）不出题。
- 返回契约显式化：delta 用稠密 `[out, in]` 交付（上游中间态是 COO 张量，稠密化即 `to_dense` 语义）；`y` 显式返回。
- 不考 `safe_merge` NaN 检查、`unmerge` 减回、量化后端 `supports_merge` 分支、`_warn_once_about_module_hooks`。
- `scaling` 钉死非负有限：上游允许任意推理期缩放，负值无实际意义且会与「值可为负」混淆，故收紧。
