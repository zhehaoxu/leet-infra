# 参考做法：调制矩阵 ⊙ 与乘性合并

## 实现

1. 校验顺序：`fan_in_fan_out` 是 bool → weight/A/B 二维有限 → r 与形状（A `[r,in]`、B `[out,r]`，由 B 的行数与 A 的列数反推 out/in）→ x 至少一维、末维 = in → fio 与 weight 形状的方向匹配。x 零维（`torch.tensor(1.0)`）也要拒。
2. **数学权重与存储布局分离**：`w_math = weight`（fio=False）或 `weight.T`（fio=True），之后所有数学运算都在 `[out, in]` 上做，最后统一转回存储布局。上游 `forward` 里写的是 `transpose(weight, fio) * _prod_AB.T`，其中 `_prod_AB = A.T @ B.T = (B@A).T`——展开后与 `w_math ⊙ (B@A)` 完全一致。
3. **合并语义**：上游 `_merge` 是 `weight.data + base_weight * new_delta`，`base_weight` 是合并前的快照（即当前 W），故 merged = `W + W⊙Δ`。写成 `W + delta`（加性）或 `eff`（丢基座）都是错的——这两颗雷都进了变异测试。
4. `hira_result = x @ eff.T`。用基座权重算（`x @ W.T`）是另一颗雷：HiRA 的残差走的是**有效权重**，不是 base 前向。

## 正确性检验

6 个 `check_*` 覆盖：题面手算示例（2×3、r=2 全部四个键的逐元素值）、4 组随机固定种子对拍（torch oracle：`x @ (W*(B@A)).T`，fio False/True 各半 + 批量 x）、fio=True 布局往返（同数学权重两种存储下 hira_result 相等、三个权重键互为转置）、一维 x、乘性 vs 加性的 W=0 锚点、merge 语义锚点（W=2I 时 merged=4I）、10 条非法输入。

oracle 是 torch 直接重算（`B@A`、`W*delta`、`x @ eff.T` 三个式子各自独立写在测试里），与本实现走同一数学但不同代码路径；测试另有逐元素手算示例兜住语义。

变异测试（5/6 全抓 + 1 个等价变异）：`⊙`→`+`（乘性变加性，示例与锚点全抓）、merged 改 `W+Δ`、merged 丢基座、fio 转置丢失、残差用基座权重。`if r < 1` 校验删除幸存——r 由 `A.shape[0]` 反推，形状校验已兜住空矩阵，该变异在可观察行为上等价（Python 下 `B@A` 对空 A 不抛错但 delta 语义无意义，形状校验先行拒绝），已在 notes 记录。

## 范围与来源

上游 huggingface/peft `src/peft/tuners/hira/layer.py`：

- `HiraLayer._merge`（L85-104）：`new_delta += get_delta_weight(...)`，`merged_weight = weight.data + base_weight * new_delta`——乘性合并的出处；`_get_base_weight_before_merge` 快照语义（本题 W 即快照）。
- `Linear.forward`（L245-273）：`_prod_AB = torch.mm(hira_A.T, hira_B.T)`，`F.linear(dropout_sub, transpose(weight, fio) * _prod_AB.T)`——有效权重 = 数学权重 ⊙ (B@A)。
- `Linear.get_delta_weight`（L213-243）：`transpose((weight_B @ weight_A), fan_in_fan_out)`——delta_weight 的布局约定。
- `reset_hira_parameters`（L149-165）：A Kaiming/高斯、B 置零的初始化（B=0 ⇒ 合并后恒等），训练侧细节不出题。

本题的口径差异：

- **不跑模型、无 dropout**：上游 forward 有 `hira_dropout(x)`（推理为 Identity），本题输入 x 直进；RoPE/attention 接线、Embedding/Conv 变体不出。
- 返回契约显式化：四个键全返回（上游只隐式持有 delta/effective）；布局钉死为「与输入 weight 相同的存储布局」。
- `fan_in_fan_out` 只考 Linear 侧的 Conv1D 布局（`[in, out]` 存取与转置）；`safe_merge` 的 NaN 检查、多 adapter 累加（`new_delta += ...`）不出。
