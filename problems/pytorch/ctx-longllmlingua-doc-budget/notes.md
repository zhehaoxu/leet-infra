# 参考做法：排序方向、预算原值传递与展平块号

## 实现

1. 校验：文档非空、`doc_ppl` 等长有限、`token_loss` 总长对齐、`context_budget` 正则、枚举白名单、force 下标界内。
2. **排序**：`direction = -1 if condition_in_question == "none" else 1`，按 `direction * ppl` **稳定**排序。上游 `get_distance_longllmlingua` 的分数是 `get_condition_ppl(d, query + " We can get the answer to this question in the given documents.") - dl * 2 / 250 * 0`——长度项乘 0，纯排序分；本题作 `doc_ppl` 入参。
3. **预算**：`target<0 → 100`，再按 `context_budget` 首字符做一次四则（上游是 `eval("target_token" + context_budget)`，本题收窄为正则白名单）。**关键**：选文档循环里扣减的变量与窗口阶段用的 `tau` 分子是**两个值**——上游 `compress_prompt` 把求值后的原 `target_token` 原封不动传给 `iterative_compress_prompt`，`control_context_budget` 内部的扣减只存在于局部变量。本题解法里 `budget`（扣减、只管停机）与 `target_token`（进 tau）分开。
4. **选文档**：按排序序扣**原始长度**（不带 +2；+2 只进窗口账），`force_context_ids` 预置、后续命中去重；`budget < 0` 即停（已扣仍入选）。`two_stage` 按**位置**奇偶切分选中序再反转奇数段——切的是 used 列表的下标奇偶，不是文档下标奇偶。
5. **动态比率**：`[i * λ/(N-1) for i in range(-(N-1), N, 2)][::-1]`，N=1 时 0；`abs(λ)`；按文档下标建 map 后跟文档重排进输出序。λ=0 直接全 0（与 map 路径数值一致）。
6. **窗口计划**：`L = len + 2`（与选文档的原始长度不同口径），`tau = budget / (ΣL + 1)`，`r = clamp(tau + dr, 0, 1)`；`last` 从 1（虚拟 BOS）起，`last + L >= W` 收窗 `(W - last, r)`，余量 `last + L - W` 整除出 k 个 `(W, r)` 整窗、余数进下一窗头，余量归零不留。
7. **阈值与掩码**：展平第 g 个 pair 覆盖 `[gW, gW + delta_end)`（上游 `end` 每 pair `+= iterative_size` 推进，推导见下）；阈值在**全流** loss 上过滤 10000 哨兵后取降序第 `clamp(int(len*ratio)-1)` 位，`ratio==1.0 → -inf`；掩码块内 `loss > thr`、位置 0 恒真、块外恒真。

**展平块号的推导**（本题最隐蔽的一点）：`get_compressed_input` 的过滤区间是 `[end_arg - delta_end, end_arg)`，其中 `end_arg = end - W + delta_end`，故区间恒为 `[end - W, end - W + delta_end)`；`end` 初值 `min(W, N)`、每 pair 末尾 `end += W`，于是第 g 个 pair 的区间恰为 `[gW, gW + delta_end)`——**按 pair 计数，不按窗口计数**（多 pair 窗口里各 pair 落在连续的不同块）。注意这是首轮、单流语义：上游真实执行时跨窗有流收缩（`end -= 已删数`）与 `past_loss` 重排，本题只做静态计划，notes 末尾写明差异。

## 正确性检验

6 个 `check_*` 覆盖：题面示例（three-layer 全链路：rank/budget/two_stage/比率/窗口/阈值/掩码逐项断言）、8 组多旋钮对拍（question 方向、force、`*2`/`/5` 预算、负 target 归 100、哨兵、N=1 比率 0、单文档 ratio 撞 1.0、5 文档 two_stage）、预算耗尽与 force 占预算、超长文档的 k 整窗 + 余数窗、阈值方向与哨兵（等值丢弃、哨兵保留但不进统计）、18 条非法输入。

oracle（`test_cases.py::_oracle`）用不同机制重写：numpy `argsort(kind="stable")` 排序、`operator` 字典做四则、**按「剩余容量」模拟窗口**（`capacity = W - last` 视角）替代上游 `last` 游标、numpy 排序取阈值。

变异测试（10/10 全抓）：排序方向翻转、预算归 100 旁路、tau 用扣减后余额、长度去掉 +2、比率表不反转、比率不按文档回填、阈值升序、保留方向翻转、块号按窗口计（多 pair 窗口漏检的关键变异）、two_stage 不反转奇数段（需要 ≥2 个奇数位选中文档才能抓到，故对拍里专门放 5 文档全选 case）。

## 范围与来源

上游 microsoft/LLMLingua（2310.06839，LongLLMLingua 集成在主仓库）：

- `prompt_compressor.py::get_rank_results`（L1818 起）与 `get_distance_longllmlingua`（L2032-2045）：`sort_direct = -1 if condition_in_question == "none" else 1`，稳定 `sorted(enumerate(...))`。
- `control_context_budget`（L1173-1241）：`eval("target_token" + context_budget)`、按排序序扣 `context_tokens_length`（原始长度）、`force_context_ids` 预置、`reorder_context` 的 original/two_stage、动态比率表与 `dynamic_ratio_map` 回填。
- `get_dynamic_compression_ratio`（L1023-1065）：`L = get_token_length + 2`、`tau = target/(Σ+1)`、`last` 游标、k 整窗扩张。
- `get_estimate_threshold_base_distribution`（L1508-1521）：`ratio==1.0 → -inf`、`ppl[ppl != 10000]`、`int(len*ratio)-1` 截断、降序取位。
- `get_compressed_input`（L1402-1506）：`need_idx = loss > threshold`（保高 NLL）、`need_idx[end:] = 1` / `need_idx[:end-iterative_size] = 1`（只动当前窗）、BOS 位移拼接 `loss[:1] > 0`。

本题的口径差异：

- **不跑模型**：文档条件 PPL 作 `doc_ppl`、逐 token NLL 作 `token_loss`（上游 loss 按下一 token 预测对齐、长度 N-1，本题对齐到 token 本身、长度 N，BOS 恒保留直接写成「位置 0 恒真」）。
- 拼接流**不插 `\n\n` 分隔符**（上游 `"\n\n".join(context)` 后整体分词，分隔符会占 token；本题文档间直接拼接）。
- `context_budget` 不用 `eval`，收窄为 `^[+\-*/]\d+(\.\d+)?$` 的单次四则；`force_context_number` 不收（上游该停机分支因 `res` 从未在循环内填充而是死代码，见 L1202 与 L1216-1219，不把死代码搬进考题）。
- **静态计划语义**：上游真实执行中，pair 的阈值算在「已收缩流」的 `past_loss` 上、跨窗有 `end` 回退与 KV 剔除；本题阈值一律算在**原始流**上、掩码也打在原始下标上，即每对 pair 相互独立的首轮语义。多 pair 的叠加收缩不在范围。
- `condition_compare` / `keep_split` / `start` / 结构化分段（`get_structured_dynamic_compression_ratio`）不在范围。
