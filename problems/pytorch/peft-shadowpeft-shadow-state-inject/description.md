# ShadowPEFT：用影子状态差值注入与门控更新跨层协调适配

LoRA 式 PEFT 把适配分散成各 Linear 独立的低秩微扰：层与层之间没有显式的任务状态被携带、更新、复用。ShadowPEFT（arXiv:2604.19254，官方 ShadowLLM/shadow-peft，已入 HF）改为**中心化的层空间精炼**：一个跨层共享、随深度演化的影子状态 `s` 与冻结骨干逐层耦合——注入时筛差值信号、编码后门控吸收新信息。

## 两个模块（官方 `modules.py`）

**ShadowInjectionModel**（应用于除 layer 0 外的每层 base decoder 之前）：

```python
delta = hidden - shadow            # (b, t, d) 差值携带偏离方向
delta_t = delta @ W_down           # W_down: (d, k), N(0, 0.02²)
delta_t = delta_t @ W_up           # W_up: (k, d), 初始全零
hidden = hidden + alpha * delta_t
```

差值先行是关键归纳偏置：瓶颈不凭空造扰动，而是在"base 与影子参考的差距"里筛有用分量。`W_up` 全零初始化保证训练起点零增量（与 LoRA 同约定）。

**ShadowUpdateModel**（每层 base 编码后更新影子状态）：

```python
h_in = LayerNorm(hidden)                     # 共享 LN
g  = sigmoid(Linear(SiLU(Linear(h_in))))     # 门控，全无 bias
ht = Linear(SiLU(Linear(h_in)))              # 变换，无 sigmoid，dropout 在中段
shadow = shadow + g * (ht - shadow)          # 等价 (1-g)*shadow + g*ht
```

GRU 风格门控 `g` 决定影子状态每层"被新信息冲刷多少 vs 保留多少任务上下文"，官方动机是防止 shadow collapse、稳定优化。**与论文公式的一处出入以代码为准**：`T` 无 sigmoid（论文写作 t=T(h_out)，实现中 T 只是两层 MLP），sigmoid 只在门 G。

## 参数量账

```
injection = L·(d·k + k·d)
update    = L·2·(d·gh + gh·d) + 2d    # 2d 是共享 LayerNorm 的 weight+bias
```

## 与 LoRA 的本质差别

影子模块与骨干**架构解耦**：可拆卸（shadow-only 推理，边缘部署）、可预训练（0.5B 小模型作为 8B 大模型的适配器）、跨层共享参数。判题域取注入/更新两个纯函数核心：差值 einsum 链、门控代数恒等式、`W_up` 零初始化不变量、参数量与 RNG 回放。
