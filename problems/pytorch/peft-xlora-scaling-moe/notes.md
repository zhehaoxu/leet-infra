# 详解：X-LoRA 缩放分类器驱动的 LoRA 专家混合

## 出处

X-LoRA（arXiv:2402.07148，Eric Buehler 等，mistral.rs 作者）已进 HF PEFT（`src/peft/tuners/xlora/`：classifier.py + model.py + layer.py）。论文口径：预训练 LoRA 适配器全冻结，缩放分类器（depth-1 单层 Linear 或 depth-d Linear+ReLU 链）从基模型 hidden states 预测 \(\Lambda \in \mathbb{R}^{s\times l\times n}\)；推理双前向（scaling pass 把缩放钉在常数上禁用专家跑基模型取 hidden states → 分类器出缩放 → forward pass 按缩放混合）。PEFT 版把 topk/clamp/global weight 的应用账放在 `layer.py::XLoraLayer`（xlora 原仓在 classifier 里）。

## 官方实现的逐行翻译

**分类器（`classifier.py`）**：

```python
class TemperatureScaledSoftmax(nn.Module):
    def forward(self, logits):
        return self.softmax(logits / self.temperature)     # dim=-1

# 层链（depth>1）:
layers = [Linear(h, xlora_size, bias=True), ReLU(), (Dropout?)]
for _ in range(depth - 2): Linear(xlora_size, xlora_size, bias=True), ReLU(), (Dropout?)
last = Linear(xlora_size, n_classes * n_layers, bias=True)   # layerwise=True
# 或 Linear(xlora_size, n_classes) + expand(-1,-1,n_layers,-1)  # layerwise=False

scalings = logits.reshape(batch, seq, n_layers, n_classes)   # C-order: (n_layers, n_classes)
if enable_softmax: scalings = temperature_softmax(scalings)

make_dummy_scalings: torch.full((b, s, n_layers, n_classes), override_scaling_pass_value)
# _set_override_scaling_pass_value(None) -> 1/n_classes
```

**每层应用（`layer.py::XLoraLayer`）**：

```python
def get_maybe_topk_scalings(self, scalings):
    xl = scalings[:, :, self.layer_number, :]
    if self.config.top_k_lora is not None:
        _, topk_indices = torch.topk(xl, k=self.config.top_k_lora, dim=-1)
        mask = torch.zeros_like(xl, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        xl = xl * mask.to(xl.dtype)                    # 原值掩码，不重归一！
    if self.config.enable_softmax_topk:
        nonzero = xl != 0
        full = xl.masked_fill(~nonzero, float("-inf"))
        new = torch.softmax(full, dim=-1)
        xl = new.masked_fill(~nonzero, 0.0)            # 全零行护栏：NaN -> 0
    return xl

@staticmethod
def apply_scalings_to_x(x, scalings_layer, adapter):
    return x * scalings_layer[:, :, adapter].unsqueeze(-1)   # 乘输入侧！

# forward:
for adapter_n, active_adapter in enumerate(active_adapters):    # 序号 = 专家列
    x_mod = apply_scalings_to_x(x, xlora_scalings, adapter_n)
    result += lora_B(lora_A(dropout(x_mod))) * scaling * global_scaling_weight
```

## 易错点

1. **topk 掩码保留原值**：被选中的 top-k 位是原始缩放值（和可能 < 1）；只有 `enable_softmax_topk=True` 才重归一。两阶段是**串联**（先掩码再 softmax 非零位），不是「softmax 后 topk」。
2. **全零行 NaN 护栏**：softmax 对全 -inf 行产生 NaN；`masked_fill(~nonzero, 0.0)` 把全零行（topk 后无任何幸存位）修成全 0。测试钉全零行过 softmax_topk 必须 finite——这是 masked_fill 唯一可观测时刻（其余行 -inf 位 softmax 本身就是 0）。
3. **缩放乘输入侧**：λ 乘在 LoRA 专家的**输入** x 上（不是输出上）。线性下数学等价（\((x\lambda)M^{\top} = x(\lambda M)^{\top}\) 逐 token 列缩放），但实现位置决定 dropout/其他算子的作用顺序；einsum 互证钉住。
4. **adapter 序号即专家列**：`for adapter_n, ... in enumerate(active_adapters)` 的循环序号直接当 `scalings_layer[:, :, adapter_n]` 的列下标——缩放矩阵的 n 维顺序与适配器激活顺序一致。
5. **scaling pass 语义**：dummy 缩放默认 0（config `scaling_pass_value=0`），None → \(1/n\)（论文的 \(n^{-1}\) 消融，结论是两者性能相近）。
6. **global_scaling_weight 只乘 LoRA 路径**：gw 翻倍 ⇒ 输出 = base + 2·(delta)；base 不受影响。
7. **layerwise reshape 顺序**：last Linear 输出 `n_classes*n_layers`，`reshape(b,s,l,n)` 是 C-order——W3 的输出维排布是 (l0c0, l0c1, l0c2, l1c0, …)（层优先）。转置重排是错位变异（m-cf-reshape 钉）。
8. **scalings=None 分支**：退化普通 LoRA（无缩放、无 global weight、sw=1）——分类器关闭/禁用 adapter 时的行为。

## 对拍口径

锚点由沙箱 dump：F1 depth-3 layerwise 分类器全链（温度 1.7）+ dummy 两种 + softmax 行性质；F2 非 layerwise expand（各层同分布）；F3 双专家层应用（dense 逐项和互证 + einsum 等价 + topk 原值掩码 + softmax_topk 非零重归一 + one-hot + 全零行 NaN 护栏 + layer_number 切片）。互证：\((x\lambda)M^{\top}=\mathrm{einsum}\)、gw 只乘增量。

## 复杂度

分类器每 token \(O(h\cdot s_{xlora} + s_{xlora}^2\cdot(d-2) + s_{xlora}\cdot n\cdot l)\)；每层应用 \(O(b\cdot s\cdot (in\cdot r + r\cdot out))\) 与普通 LoRA 同阶（缩放是逐元素乘）；双前向使推理成本约 2 倍（第一遍可 no_grad 且禁用专家）。
