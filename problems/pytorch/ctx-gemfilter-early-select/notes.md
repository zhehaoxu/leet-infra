# 参考做法：interleave 展开、最后一个 query 与「求和 + 零填充均值池化」

## 实现

1. 校验嵌套形状（各头 token 数一致、维度一致、Q/K 同维、`num_heads == kv_heads * groups`）、`topk >= 1`、`kernel_size` 正奇数、`norm > 0`、所有值有限。
2. **GQA 展开**用 `repeat_interleave(groups, dim=0)`——transformers `repeat_kv` 的 expand+reshape 等价于沿头维 interleave；错用 `repeat` 是本题埋的第一颗雷。
3. **最后一个 query**：`queries[:, -1:, :]`（上游 `query_states[:, :, -1:, :]`，去掉 batch 维后即最后一行）。改前面位置的 Q 不许影响结果。
4. **内积 + 头间求和**：`matmul(q_last, K^T)[:, 0, :]` 得 `[H, n]`，`sum(dim=0)` 收成 `[n]`。上游 `sum_over_heads=True` 走的就是求和分支（`sum(dim=1, keepdim=True)`）；不开时每个头各自 top-k，本题不做该分支。
5. **均值池化**：`F.avg_pool1d(sums[None,None,:], kernel_size, padding=kernel_size//2, stride=1)`。两个默认值是考点：padding 补 **0**、`count_include_pad=True` 故**分母恒为 kernel_size**——边界窗口的均值天然被压低，这不是「按实际窗口长度平均」。
6. **top-k**：`k_eff = min(topk, n)`（上游传给 `torch.topk` 前先 clamp，k > n 会直接抛错）；排序键 `(-pooled[j], j)`：值降序、**同值下标升序**（torch.topk 对同值的次序未定义，本题把契约钉死为下标升序；测试数据的整数内积保证边界不出现同值歧义，同值只出现在已同选/同弃的内部）。`norm != 1` 时 `distances` 除以 `norm`。

## 正确性检验

6 个 `check_*` 覆盖：题面示例（手算 sums、pooled、两种 topk、norm 变体）、4 组随机固定种子对拍（MHA 8 头、GQA 4×3、kernel=1 恒等池化、负值混排 + norm=0.5）、interleave 与 repeat 的区分（手算两种 sums）、groups=1 退化 MHA 的手算、非末位 query 不敏感性、池化边界零填充（含「分母恒为 kernel_size」的手算）、18 条非法输入。

oracle（`test_cases.py::_oracle`）用纯 Python float64 独立重写（显式 interleave 循环、逐点内积、边界裁剪求和），与 torch float32 路径只在浮点误差内一致；测试数据用小整数保证排序不受精度影响。

变异测试（6/6 全抓 + 1 个等价变异）：`repeat_interleave`→`repeat`（全部用例抓，TypeError 之外数值全变）、最后一个 query 换成第一个、头间求和换成取第 0 头、池化改 `count_include_pad=False`、tie-break 下标降序、`distances` 不除 norm。`k_eff = min(topk, n)` 去掉 clamp 在 Python 切片下与原实现等价（切片越界自动截断），不是有效变异——clamp 保护的是上游 `torch.topk(k>n)` 的抛错路径，已在 notes 记录。

## 范围与来源

上游 SalesforceAIResearch/GemFilter（2409.17422）：

- `my_baseline/GemFilter/gem_filter_utils.py::standard_dis_index`（L6-19）：`torch.matmul(queries, data.transpose(-1,-2))` → 取 `[:, :, 0, :]` → `sum_over_heads` 时 `sum(dim=1, keepdim=True)` → `pool` 时 `F.avg_pool1d(kernel_size, padding=kernel_size//2, stride=1)` → `torch.topk(k, dim=-1)` → `norm != 1` 时 `distances / norm`。
- `gem_filter_utils.py::find_context`（L22-34）：仅在 `select_layer_idx` 层、`self.indecies is None` 时打分；`repeat_kv(key_states, num_key_value_groups)`、`query_states[:, :, -1:, :]`、`k = min(self.topk, n)`、`pool=True, sum_over_heads=True`（kernel_size 未传即默认 5）。
- `llama_select_attention.py`（transformers 4.43.3 口径）：`topk = 1024`、`select_layer_idx = 13`；select 阶段不做 KV cache 更新，索引存下后供后续层裁剪输入。

本题的口径差异：

- **不跑模型、不接 flash attention**：RoPE 后的 K、Q 与分组数全部显式入参；b=1（上游 `assert b == 1`）。
- 返回契约显式化：`indices` 钉死「值降序、同值下标升序」，`pooled` 全量返回（上游只返回 top-k 的 values/indices）。
- `norm` 参数保留（上游默认 1，未传即不除），`sum_over_heads=False` 与 `pool=False` 分支不出题。
- 只做打分与选择；「选中索引如何喂给后续层的 KV 裁剪」涉及 cache 接线，不在范围。
