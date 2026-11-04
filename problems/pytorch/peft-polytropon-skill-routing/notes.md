# 解题笔记 — Polytropon

## 官方账

- HF PEFT `peft/tuners/poly/layer.py`（PolyLinear）+ `router.py`（PolyRouter）；论文 Polytropon + MHR。
- 判题域：初始化 RNG 域、推理路由、einsum 混合、1/r 缩放前向、参数账。RelaxedBernoulli 训练分支不在域内。

## 踩坑

- **kaiming 浮点公式**：`math.sqrt(5)**2 = 5.000000000000001`，bound 与精确 `1/√fan_in` 差 ~1e-16，uniform 抽样位级不同——对拍必须复用同一公式（`1.0/math.sqrt(d)` 会挂）。
- **循环顺序即 RNG 域**：skill 外层、split 内层；router logits 在所有 A 块之后抽。`init_weights=False` 时 B 块继续同一 generator 流。
- **B 初始化参数序**：官方 `empty((d_out, r))` kaiming（fan_in=r）转置成 (r, d_out)。传反成 `empty((r, d_out))` 会形状错（RuntimeError 兜住，但语义是 fan_in=d_out 的错误分布）。
- **路由不是 softmax**：sigmoid 后按 skill 维归一化；EPS=1e-12 在 sigmoid 下溢（logits≈-800）时防 NaN——数值域内 EPS 不可见，但极小 logits 用例可判题。
- **组合 reassembly**：A reshape 是 split-major 输入列；B 需要 transpose(1,2) 再 reshape（漏 transpose 的变异被形状/数值抓）。
- **缩放严格 1/r**：不是 α/r 或 1/√r（LoRA 惯性思维易错）。
- 锚值占位教训：先写占位锚值再跑真值填充时，FAIL 报的值就是真值（本树两次直接回填成功）；但必须确认 solution 本身正确（有 manual 对拍兜底）再回填。

## mut 17 变异清单

init：循环序互换（skill/split）、B zeros→ones、A/B kaiming 参数序（fan_in 换位）、router range 扩大 10 倍；route：去 EPS、softmax 替换、view(ns,sk) 互换、越界边界 >=→>；combine：A 混合系数、B 漏 transpose、A reshape 转置；delta：1/r→1/√r、去缩放、bmm 交换（形状错）；param_count：router 公式、lora +1。17/17 全 CAUGHT。
