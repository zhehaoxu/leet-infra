# 详解：PVeRA 概率向量随机矩阵自适应

## 出处

PVeRA（Probabilistic Vector-based Random Matrix Adaptation，arXiv:2512.07703，Fillioux 等）已进 HF PEFT v0.20.0（`src/peft/tuners/pvera/`）。VeRA 证明「跨层共享冻结随机 \(A,B\) + 每层两个小向量」可用 1/10 参数量逼近 LoRA；PVeRA 把每层潜在增量从定值 \(\lambda_d \odot (xA^{\top})\) 升级为分布——\(\lambda_d\) 变长 \(2r\)（前半缩放 μ 通路、后半缩放 logvar 通路），前向走重参数化 \(z=\mu+\epsilon\,e^{\frac12\log\sigma^2}\)，训练后可采样出校准的置信区间（VTAB-1k 上优于 VeRA）。

## 官方实现的逐行翻译

**共享银行（`model.py::_init_pvera_A_pvera_B`）**：

```python
generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
pvera_A = torch.nn.init.kaiming_uniform_(torch.empty(config.r * 2, linear_in_dim), generator=generator)
pvera_B = torch.nn.init.kaiming_uniform_(torch.empty(linear_out_dim, config.r), generator=generator)
```

注意：`kaiming_uniform_` 默认 `a=0`（不是 `nn.Linear.reset_parameters` 的 \(a=\sqrt5\)）→ bound \(=\sqrt{6/\text{fan\_in}}\)。**\(A\) 有 \(2r\) 行**（μ/logvar 双份）、\(B\) 只有 \(r\) 列。多层不同形状取 max（`_find_dim`），每层 forward 切片。

**层内（`layer.py`）**：

```python
self.pvera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features))   # 全 1！
self.pvera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r * 2))              # reset 时填 d_initial
# reset_pvera_parameters:
nn.init.zeros_(self.pvera_lambda_d[...]).fill_(d_initial)                          # 全 d_initial（默认 0.1）
nn.init.zeros_(self.pvera_lambda_b[...])                                           # 先零后——注意 lambda_b 不被 reset 覆盖？
```

细读：`reset_pvera_parameters` 里 `zeros_(lambda_d).fill_(d_initial)` 只作用 lambda_d；`zeros_(lambda_b)` 把 lambda_b 清零。**但 lambda_b 在 `update_layer` 里是 `ones` 创建、init_weights=True 时立即被 reset 的 zeros_ 覆盖**——训练起点 delta=0（与 VeRA 同零起点语义）。init_weights=False 跳过 reset：lambda_b 保持全 1、lambda_d 保持 `torch.randn` 初始（未定，不可锚）。本题 `layer_setup` 入口钉 **reset 后账**（λ_b 全零、λ_d 全 d_initial）并显式返回 `lambda_b_uninit`（全一分支）。`d_initial` 只进 lambda_d。

**forward（`layer.py::forward` + `_reparametrize`）**：

```python
sliced_A = pvera_A[:, :in]; sliced_B = pvera_B[:out]
mu, logvar = (lambda_d * F.linear(dropout(x), sliced_A)).chunk(2, dim=-1)
result = result + lambda_b * F.linear(self._reparametrize(mu, logvar, sample_at_inference), sliced_B)
# _reparametrize: training or sample_at_inference -> mu + randn*exp(.5*logvar); else mu
```

**merge（`get_delta_weight`）**：

```python
lambda_d = lambda_d[:lambda_d.size(0)//2]     # μ 半段
sliced_A = sliced_A[:sliced_A.size(0)//2]
output = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), fan_in_fan_out)
```

## 易错点

1. **λ_b 的「ones 创建、reset 清零」**：`update_layer` 先 `torch.ones` 再被 `reset_pvera_parameters` 的 `zeros_` 覆盖——init_weights=True 训练起点是**零**。官方代码这里极易误读（两行相邻的 zeros_ + fill_ 只影响 lambda_d 的数值，zeros_(lambda_b) 单独清零 λ_b）。本题按 reset 后语义锚定，`layer_setup` 显式区分两分支（True→λ_b 零、False→λ_b 一）。
2. **μ/logvar 共享一个线性映射**：`chunk` 切的是 `(λ_d ⊙ xA^{\top})` 的特征维——\(A\) 上半行产生 μ、下半行产生 logvar，λ_d 前后半各缩放自己的半段。不是两个独立 Linear，也不是 μ=xAμᵀ、logvar=xAlogvarᵀ 之外再拼 λ——λ_d 是逐元素乘在 **2r 维联合输出**上再 chunk。
3. **无 scaling 项**：VeRA/LoRA 都有 \(\alpha/r\) 或 \(1/\sqrt r\) 缩放；PVeRA forward 与 merge 都没有（config 无 lora_alpha 字段）。
4. **merge 只取 μ 半段**：logvar 半段对 merge 与 eval 前向均不可见——测试钉「改 logvar 半段 eval/merge 输出不变」。
5. **eval vs 采样**：`_reparametrize` 在 `self.training or sample_at_inference` 时采样、否则直接 μ。判题锚点用 eps 显式入参钉采样路径（z=μ+eps·exp(½logvar)），eval 路径确定性。
6. **kaiming 默认 a=0**：bound=\(\sqrt{6/fan\_in}\)，float32 RNG 逐位锚定（float64 与 float32 的 uniform RNG 序列不同，不能跨 dtype 对照——沙箱实测教训）。

## 对拍口径

锚点由沙箱 dump：F1 共享银行 RNG 账（float32 逐位）+ 双层切片 + eval/采样双路径 + merge 互证 \(x@dW^{\top}\)；F2 r=1 最小秩 + d_initial=0.25；F3 均匀 λ_d 与 μ 半段全零零起点；setup 钉 init_weights 两分支（True→λ_b 零、False→λ_b 一）与 d_initial 只进 λ_d。互证：eval 前向 ≡ \(x\,\Delta W^{\top}\)、改 logvar 半段不动 eval/merge、采样路径手工重参数化逐位一致。

## 复杂度

每层前向 \(O(BN(in\cdot 2r + r\cdot out))\)；可训练参数 \(2r + out\) 每层（\(A,B\) 全层共享、冻结）——与 VeRA 同阶（lambda_d 翻倍），分布语义白拿。
