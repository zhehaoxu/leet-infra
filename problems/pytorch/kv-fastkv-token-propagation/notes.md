# 解题说明 · FastKV 跨层 token 传播

## 一句话

在 TSP 层用跨 KV head 求和的 token 分数挑一个子集，后续层只保留这个子集；同时把后续层的保留率从 `retain_rate` 放宽到 `retain_rate / tsp_rate`，使绝对预算大致不变。

## 上游回源

`dongwonjo/FastKV`（arXiv:2502.01068，README 亲见）：

- `baselines/fastkv/utils.py::FastKVCluster.update_kv`：逐层打分与保留，SnapKV 式窗口打分（`window_size=8`、`kernel_size=7`）。
- `baselines/fastkv/utils.py::compress_fastkv`：按 `retain_rate` 截断；TSP 层之后的层把预算放宽为 `retain_rate / tsp_rate`。
- `baselines/fastkv/llama_model.py::llama_decoderlayer_forward_fastkv`：TSP 层选 token，后续层复用该子集，`position_ids` 保留原值。
- 默认 `tsp_idx=15`、`tsp_rate=0.2`、`retain_rate=0.1`；本题把 `tsp_idx` 与比例都做成显式入参。

## 与既有题的边界

- `kv-snapkv-pooling`：打分同源（窗口注意力 + 池化），但本题**不实现池化**，`attn_caches` 直接给逐 token 分数；考的是跨层 TSP 传播与后续层预算放宽。
- `kv-simlayerkv`：按 0/1 懒惰判定决定某层走 full attention 还是复用上一层；本题不是二值跳层，而是在 TSP 层选出**具体 token 下标**并逐层传播。
- `kv-key-rerotation-rope`：会改写 key 的 RoPE 相位；本题不重旋转，只做分数排序与下标传播。

## 为可判做的简化

- `attn_caches` 是逐层 `(T_l, H_kv)` 的显式分数，替代上游从 q/k 现算的 SnapKV 打分。
- `hidden_states` 与 `position_ids` 显式给出，`solve` 只做 `index_select`，不跑模型前向。
- `q_len` 表示当前段长度；`tsp_idx` 显式；`window_size` / `retain_rate` / `tsp_rate` 带默认值。
- 各层 cache 长度由 gating 规则唯一决定：`l <= tsp_idx` 或 `q_len <= tsp_length` 时为 `q_len`，否则为 `tsp_length`。
- 同分取更小下标；`kept` 每行升序返回，窗口恒在尾部。
- 返回 `capacities` 与逐层 `kept`，便于检查预算和选择结果。

## 已知上游差异

- 上游打分包含 `kernel_size=7` 的池化与窗口注意力；本题分数是入参，若调用方给的分数不是上游口径，排序会不同。
- 上游 `tsp_idx=15` 固定、预算比例来自配置；本题允许任意 `tsp_idx` 与比例。
- 上游处理 decode 循环与变长 cache；本题只处理一次给定的 `q_len` 段。
- 上游可能对 GQA 的 KV head 做 group 处理；本题逐 KV head 独立保留，TSP 分数在 KV head 维求和。
- **同名论文区分**：本题对应 `dongwonjo/FastKV`（arXiv:2502.01068，training-free KV cache 压缩 + token-selective propagation）。ACL Findings 2026 的 **"FastKV: Decoupling of Context Reduction …"** 是另一篇同名工作，作者、方法与代码库都不同；不要把两者混为一谈，也不要把后者的结论写进本题。

## 复杂度

逐层排序 \(O(\sum_l T_l H_{\text{kv}}\log T_l)\)，TSP 额外 \(O(T_{\text{tsp}}\log T_{\text{tsp}})\)；相对全量 attention 的 \(O(T^2)\) 只是选择与切片的开销。
