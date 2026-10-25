# 解题说明 · Q-Filter

## 直觉：重要性是学出来的，不是猜出来的

家族里其它打分都是**手工先验**：knorm 猜"大范数重要"（order 146），observed attention 猜"被看得多重要"（149），expected attention 猜"未来 query 会看向它"（150）。Q-Filter（arXiv:2503.02812）的立场是：这些先验在 QA 精度上都有系统偏差，不如为每个 (层, kv head) 学一个滤波器 \(f\in\mathbb{R}^{D}\)，让 \(-f\cdot k\) 直接预测"驱逐这个 key 对答案的损失"。滤波器以 `QFilters`（一个 `nn.Parameter` 张量，形状 `(num_layers, num_kv_heads, head_dim)`）随模型存到 HuggingFace hub，`post_init_from_model` 时按模型名自动下载——这正是判题环境（离线、无 kvpress）必须把滤波器作为参数直接喂给 `solve` 的原因。

## 核心三行

```python
q_filter = self.q_filters[module.layer_idx][None, :, None]   # (1, H_kv, 1, D)
scores = -(q_filter * keys).sum(dim=-1)                      # (B, H_kv, k_len)
```

之后走 ScorerPress 通用流程：top-\(n\_kept\) 选择。要点：
- **负号**：分数 = 负点积。top-k 取大 ⇒ 实际取的是与 \(f\) 点积最小（最反向）的 key。
- **无缩放**：不像 QKᵀ 那样除 \(\sqrt{D}\)；排序不受常数影响，但输出 `scores` 时必须保持原始量值。
- 滤波器逐 (layer, kv head) 独立，**不跨 head/层共享**，也不做组平均（kv head 一一对应）。

## 与近邻题对比

- `kv-knorm-importance`（146）：score = ‖k‖₂，旋转/基不变；Q-Filter 依赖 key 与固定方向 \(f\) 的内积，**与坐标系强耦合**——RoPE 位置旋转会直接改变分数（这解释了 167 KeyRerotation 为什么和它是绝配组合）。
- `kv-expected-attention`（150）：也要 query 侧统计量（μ, Σ）；Q-Filter 把"未来 query 长什么样"整个折叠进一个学到的向量，输入只要 keys。
- `kv-think-channel-mask`（153）：Think 的通道掩码是**每 token 选哪些通道**参与求和；Q-Filter 是**每 head 一个固定通道权重向量**作用于所有 token——前者稀疏 mask，后者稠密内积。

## 陷阱

- 忘记负号是最大考点：结果恰为对称的反向选择集。
- 平局语义：滤波器为零、或与部分 key 正交 → 分数相等；按 (−score, position) 稳定排序，平局保左。
- \(n\_kept\) 保底 1：\(r\to1\) 时 \(\lfloor L(1-r)\rfloor=0\) 也要返回最左/最高 1 个。
- 输出 `kept` 升序（`scores` 保持原顺序不排）。
