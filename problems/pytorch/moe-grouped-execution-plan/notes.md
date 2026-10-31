# 参考做法：把槽位排成组、算完再还原

## 实现

1. `flat_ids = expert_ids.reshape(-1)`；`perm` 来自 `torch.sort(flat_ids, stable=True)` 的第二项，`sorted_ids` 是第一项。
2. `token_rows = torch.div(perm, k, rounding_mode="floor")`，然后 `hidden[token_rows]` 就是分组 kernel 的输入矩阵。
3. `gather_ids = torch.clamp(sorted_ids, max=E - 1)`，`grouped = hidden[token_rows] * diag[gather_ids]`；再用 `grouped.masked_fill(sentinel.unsqueeze(-1), 0.0)` 把哨兵行整行清掉。顺序不能反：先 clamp 是为了索引在界内，后置清零是为了让哨兵行不留下任何数值。
4. `counts = torch.bincount(flat_ids[flat_ids < E], minlength=E)[:E]`，`offsets = torch.cumsum(counts, 0)`。
5. `inv_perm = torch.empty(S, dtype=torch.int64)`；`inv_perm[perm] = torch.arange(S)`。这是「用 scatter 得到逆排列」的唯一正确写法，方向写反得到的还是 `perm`。
6. `out = (grouped * flat_weights[perm].unsqueeze(-1))[inv_perm].reshape(T, k, D).sum(1)`。

四个常见错误：

- **忘了稳定排序**。同专家内部的行序对真实 kernel 无所谓，但 `perm`、`inv_perm`、`token_rows` 三张表的具体数值完全由它决定；判题要逐位比这些表。
- **`token_rows` 用 `perm % k` 而不是 `perm // k`**：形状完全对得上，只有数值会错，很容易漏。
- **忘了哨兵清零**：clamp 之后哨兵行会拿到专家 $E-1$ 的真实输出，再乘上它自己的权重，把「被丢弃的 token」算进了结果。
- **用 `index_add_` 思路去 scatter**：本题要求 `out` 的累加顺序是 $j=0,\dots,k-1$，先 unsort 再 reshape+sum 才是这个顺序。

## 正确性检验

7 个 `check_*` 覆盖：手算示例（七个返回项全部逐位核对）、随机对拍（8 组形状/哨兵数组合 + 两种非连续布局）、计划不变量（`perm` 是排列、`inv_perm[perm] == arange`、`token_rows == perm // k`、排序后专家编号非降、尾部恰好是哨兵、`counts.sum() == offsets[-1] == valid`、每行 grouped 等于 `hidden[token] * diag[expert]`、`out` 与「不经过排序的直接按 token 累加」一致）、全哨兵与零权重、单路径（`k=1`、单 token 单专家、所有槽位同专家）、非法输入、最大形状 `T=k=8, E=8, D=4`。

oracle 是纯 Python：`sorted(range(S), key=lambda s: (ids[s], s))` 做稳定排序，列表推导生成 grouped 行，手工前缀和做 counts/offsets，`inv_perm[s]` 用循环填，最后每个 token 用 `math.fsum` 沿 $j$ 累加。完全不使用 torch 的 `sort`/`bincount`/`cumsum`/`gather`，也不做广播。

变异测试：`token_rows` 取模、不 clamp、不清零哨兵、`counts` 把哨兵也算进最后一个专家、`inv_perm` 方向写反、根本不 unsort、以及把 unsort 写成 `index_copy_` 的错误映射——七个都被抓到。不加 `stable=True` 的 `torch.sort` 在本机只在 64 槽位的最大形状用例上才暴露差异，小输入上与稳定排序巧合一致，这说明最大形状用例是钉住并列口径的必要用例，不是凑数。

## 范围与来源

索引代数取自判题环境里 transformers 5.x 的 `transformers/integrations/moe.py::grouped_mm_experts_forward`（`_grouped_linear` 的上游实现），逐行对应关系如下（省略了 gate/up 双投影与激活）：

```text
expert_ids_g, perm = torch.sort(expert_ids.reshape(-1))
selected_hidden_states_g = hidden_states[perm // num_top_k]
sample_weights_g = sample_weights[perm]
histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
expert_ids_g.clamp_(max=self.num_experts - 1)
proj_out = proj_out.masked_fill(sentinel_mask, 0.0)
weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
inv_perm = torch.empty_like(perm)
inv_perm[perm] = torch.arange(perm.size(0), device=device)
weighted_out = weighted_out[inv_perm]
final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
```

上游注释里对哨兵的说明与本题一致：「让 `expert_ids` 保持不 clamp 地参与排序，哨兵自然到尾部，`histc(max=num_experts - 1)` 把它们从 `tokens_per_expert` 里排除，grouped_mm 只算 `offsets[-1]` 之前的行；kernel 会把尾部行留成未初始化，靠一次 pre-mask 加一次 post-mask 兜住」。本题把「未初始化」显式化成「清零」，因为 CPU 上没有未初始化内存可依赖。

与上游的**刻意差异**：`counts` 用 `bincount` 而不是 `histc`（`histc` 在 CPU 上只吃浮点、在 CUDA 上只吃整数，正是上游那段 dtype 分支的原因，与本题无关）；`offsets` 返回 int64 而不是 int32；排序显式要求稳定；专家内部计算被换成一次对角乘。`batched_mm` 那条路径（把 token 按 `num_top_k` 复制后用批量矩阵乘）不在本题范围，容量式分派见 `moe-capacity-dispatch`，all-to-all 的分派与合并见 `moe-all-to-all-dispatch`。
