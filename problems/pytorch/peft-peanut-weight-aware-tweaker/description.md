# PEANuT：用权重感知神经 Tweaker 生成任务自适应更新

LoRA 用两个可训练低秩矩阵从零学近似 `ΔW`，是**权重无关**的线性近似：无论冻结权重长什么样，适配器结构都一样。PEANuT（arXiv:2410.01870）把冻结权重本身作为条件输入：一个紧凑神经网络（官方称 *tweaker*）读入 `Wᵀ` 并生成任务自适应更新，理论上表达能力不低于任何线性 PEFT，且在 `in_features > out_features` 的层（如 MLP 的 `down_proj`）参数量严格少于 LoRA。

## 官方结构（HF PEFT `src/peft/tuners/peanut/layer.py`）

以 base 层 `W: (out, in)`、秩 `r`、深度 `depth` 记：

- `peanut_A: Linear(out_features → r)`——注意输入维度是 base 的 **out_features**（不是 in_features），这是参数量优势的来源；
- `depth` 对 `encoder/decoder`，各为 `Linear(r → r)`；
- `peanut_B: Linear(r → out_features)`。

前向增量构造（`_compute_delta_weight`）：

```python
d = act(W.T @ A.T)          # (in, r) —— 权重条件化入口
res = []
for enc in encoders:        # 先存残差，再过 encoder
    res.append(d)
    d = act(enc(d))
for i, dec in enumerate(decoders):
    d = act(dec(d)) + res[depth - 1 - i]   # 残差反序相加
delta = (B(d) * scaling).T  # (out, in)
```

三个非显然细节：

1. **残差反序**：encoder 阶段 `depth` 次各存一份残差，decoder 阶段按 `res[depth-1-i]` 倒序相加——配对是 `(decoder_i, residual_{depth-1-i})`，不是栈式对称；
2. **激活逐层都有**：入口投影、每个 encoder、每个 decoder 都过 `act_fn`（默认 relu），残差加在 decoder 激活**之后**；
3. **零初始化不变量**：`init_weights=True` 时 B 全零 → 合并增量恒为全零（训练从零增量起步，与 LoRA 同约定）。

## 参数量账

```
LoRA   = in·r + r·out
PEANuT = out·r + r·out + 2·depth·r²
saving = r·(in − out) − 2·depth·r²
```

对 `down_proj`（1024→256、r=8、depth=1）：LoRA 10240，PEANuT 4224，省 6016。深度换表达力的代价是每对 encoder/decoder `2r²`。

## 初始化

`reset_peanut_parameters` 的 RNG 消耗顺序固定：`A` → `encoders[0..depth-1]` → `decoders[0..depth-1]` 各 `kaiming_uniform_(a=√5)`（即 `uniform(±1/√fan_in)`，`peanut_A` 的 fan_in 是 **out_features**），最后 B 零初始化或 kaiming。复现锚值必须按此顺序回放。
