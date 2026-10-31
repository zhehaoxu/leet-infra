# 解题说明 · TriAttention 相位打分

## 一句话

反解 RoPE 后，把 `q_mean` 与 `k_pre` 的复数乘积写成「幅值 × cos(相位)」，相位里叠加几何偏移得到的时间项；逐 `(l,h)` 在 token 维 z-normalize，跨层取均值，再逐 head top-k。

## 上游回源

`WeianMao/triattention`：

- `longlive/kv_compression.py::_compute_layer_head_scores`：主路径。
  - 解析逆转 RoPE：`k_pre = k_post * exp(-iθ)`；
  - `relative = q_mean * conj(k_pre)`，取相位 \(\varphi\)；
  - `phase = Δt·ω·temporal_mask + φ`；
  - \(\Delta t\) 取几何偏移 `1, 2, 4, …`；
  - `score = Σ_f amp·cos(phase) + Σ_f extra`；
  - 逐 `(l,h)` 在 token 维 z-normalize，再对层取均值；
  - 预算常量 `KV_BUDGET=2048`、`DIVIDE_LENGTH=128`、`WINDOW_SIZE=128`；
  - 对 GQA 显式 `raise`。
- arXiv：README badge 标注 `2604.04921`；本次只在 badge 见到编号，未打开论文正文，因此题面按「README badge」引用。

## 与既有题的边界

- `kv-key-rerotation-rope`：把 KV 的 key 重新旋转到统一 RoPE 坐标系，会**改写 key**。本题只借用「反解 RoPE」来算相位分数，**不重旋转、不写回 key**，输出的是分数与 token 下标。
- `kv-expected-attention`：从 q/k 的统计量估计期望注意力并据此选 token。本题是复数相位 / 幅值的几何打分，再 z-normalize、层均值，打分公式与它不同。
- `kv-snapkv-pooling`：窗口注意力 + 一维池化的逐 key 分数；本题是逐频率的相位项求和，不涉及池化。

## 为可判做的简化

- `q_mean` / `q_abs` / `k` / `rope_cos` / `rope_sin` / `omega` / `pos` 全部显式入参；上游从模型 hidden states 与 RoPE cache 中取。
- `keep` 是显式预算，替代 `KV_BUDGET` / `WINDOW_SIZE` 的 eviction 流程；`temporal_count`、`offset_max_frames` 也显式给出。
- `normalize_scores` 可关闭，便于检查原始 `score`；`score_aggregation` 可切 `mean` / `max`。
- 返回 `phi` / `delta_t` / `score` / `normalized` / `head_score` / `kept` 全部中间量，便于逐项断言。
- 输出 `phi` 从内部的 `(L,T,H,F)` permute 成 `(L,H,T,F)`；`delta_t` 以 float 返回（形状 `(T,O)`）。
- 同分取更小 token 下标；上游用 `tie_break_noise` 抖动打破平局。

## 已知上游差异

- 上游的 `KV_BUDGET=2048` / `DIVIDE_LENGTH=128` / `WINDOW_SIZE=128` eviction 流程没有实现；本题只做打分与显式 top-k。
- 上游对 GQA 显式报错；本题要求 `q_mean` 与 `k` 的 `H` 一致，不建模 group 复制。
- 上游 RoPE 张量的实际布局可能与本题的「相邻实虚对」打包不同；本题把 `k[..., 2f]`、`k[..., 2f+1]` 视为一对。
- `q_abs` 在上游是逐 `(l,h,f)` 的查询幅值统计；本题直接给值，`q_abs == |q_mean|` 时 `extra` 为 0。
- 上游 z-normalize 后按层均值再 top-k，与本题一致；差异主要在预算、tie-break 与中间量的暴露方式。

## 复杂度

设偏移数为 \(O\)（默认 128 帧对应 8 档），打分约 \(O(LTHFO)\)，z-normalize 与 top-k 为 \(O(LHT + HT\log T)\)；相对全量 attention 的 \(O(T^2)\) 是线性开销。
