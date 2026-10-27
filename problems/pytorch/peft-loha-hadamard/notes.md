# PEFT：LoHa Hadamard 积低秩适配——算法详解

## 1. 公式与出处

LoHa 出自 FedPara 系列（Hyeon-Woo et al., *FedPara: Low-Rank Hadamard Product for Language Models*, arXiv:2108.04916），PEFT 库在 `tuners/loha/layer.py` 中实现。Linear 层的参数命名与形状（`create_adapter_parameters`）：

- `hada_w1_a`: \((d_\text{out}, r)\)，`hada_w1_b`: \((r, d_\text{in})\)
- `hada_w2_a`: \((d_\text{out}, r)\)，`hada_w2_b`: \((r, d_\text{in})\)

`make_weight`（`HadaWeight.forward`）只有一行核心：

```python
diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale      # scale = alpha / r
```

即 \(\Delta W = s\,(W_{1a}W_{1b})\odot(W_{2a}W_{2b})\)，缩放与 LoRA 同款 \(s=\alpha/r\)（`update_layer` 中 `self.scaling[adapter_name] = alpha / r`）。合并（`get_delta_weight` 后 `weight += delta`）就是本题的 \(W' = W_0 + \Delta W\)。

## 2. 为什么乘法能"满秩"

LoRA 的 \(\Delta W = s\,BA\) 恒有 \(\operatorname{rank}(\Delta W)\le r\)。而 Hadamard 积有著名的秩性质（Matousek & Stivner）：

\[\operatorname{rank}(A\odot B)\ \le\ \operatorname{rank}(A)\cdot\operatorname{rank}(B)\]

两条秩-\(r\) 分支相乘，秩上界变成 \(r^2\)，且一般位置下可以取到——也就是用 \(2r(d_\text{out}+d_\text{in})\) 个参数表达满秩 \(d_\text{out}\times d_\text{in}\) 更新。FedPara 正是用它做到「参数量不超过 LoRA、表达力却是满秩」。反例断言（`check_hadamard_not_sum`）对应的直觉：乘法有"逐元素门控"效应，位置级相关性是加法给不出来的。

## 3. 初始化与退化（判题考点）

- PEFT/LyCORIS 的默认初始化（`reset_adapter_parameters`）把 `w2_b` 置零、其余 Kaiming——起点处 \(\Delta W=0\)，适配器"什么都不做"（注释原话：*It is enough to initialize only single matrix with zeros to make adapter do nothing*）。`check_zero_branch_degenerates_to_base` 验的正是它。
- \(\alpha=0\) 显然 \(\Delta W=0\)；\(r=1\) 时两分支各是一次外积，\(\Delta W = s\,(u_1v_1^\top)\odot(u_2v_2^\top) = s\,(u_1\odot u_2)(v_1\odot v_2)^\top\)，退化为一条秩 1 的 LoRA——乘法在秩 1 时与加法同阶。
- Conv 层的 LoHa 还有 t 因子重建（`make_weight_cp`），本题只考 Linear 形态。

## 4. 与本合集其他题的分工

- `peft-lora-forward`/`peft-lora-merge`：加法组合 \(s\,BA\)，秩 \(\le r\)。
- `peft-dora`：在 LoRA 上做幅值-方向分解。
- 本题：乘法组合，\(\Delta W\) 可满秩；注意形状约定是 PEFT 源码的 \((d_\text{out},r)\times(r,d_\text{in})\)，与 LoRA 题的 \(A\in(r,d_\text{in}),B\in(d_\text{out},r)\) 命名相反、本质同构。
