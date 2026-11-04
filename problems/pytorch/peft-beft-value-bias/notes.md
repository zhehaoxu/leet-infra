# 详解：BEFT 旁路偏置高效微调

## 出处

论文 BEFT: Bias-Efficient Fine-Tuning of Language Models in Low-Data Regimes（arXiv:2509.15974，ACL 2026 主会，Baichuang Huang / Ananth Balashankar / Amir Aminifar，隆德大学 + Google DeepMind，已入 HF PEFT v0.20 `tuners/beft`）。研究问题：BitFit 证明「微调全部 bias」在低数据下能逼近全参微调，但注意力中 \(b_q/b_k/b_v\) 三类 bias 并不等价——**哪类最值得训？**结论：直接微调 \(b_v\) 在低数据场景一致优于 \(b_q/b_k\)，覆盖 encoder-only/decoder-only 至 6.7B（含无 bias LLM）。

结构性论证：softmax 注意力 \(Q=XW_q+b_q\)、\(K=XW_k+b_k\)、\(V=XW_v+b_v\)——给每个 Key 同加 \(b_k\) 使所有 attention logit 平移同常数，被 softmax **抵消**；\(b_q\) 的效果依赖 query-key 交互；\(b_v\) 直接线性作用于 value 通路，每比特梯度都有效。PEFT 实现层面，BEFT 对目标层**新增**旁路 bias（而非训练原 bias——无 bias 模型也能用），这是与 BitFit 的实现分界。

## 官方实现的逐行翻译

`layer.py` 全文三段账（去设备/dtype 细节）：

```python
# update_layer: 旁路 bias 形状账 + 零初始化
weight = torch.randn((1, self.out_features))
self.beft_bias[adapter_name] = nn.Parameter(weight)
if init_weights:
    nn.init.constant_(self.beft_bias[adapter_name], 0.0)
# 注: base bias 为 None 时仅警告「不能 merge」，forward/训练照常

# forward: 多 adapter 先求和再一次加（不改原层）
beft_bias = None
for active_adapter in self.active_adapters:
    current_bias = self.beft_bias[active_adapter].flatten()
    beft_bias = current_bias if beft_bias is None else beft_bias + current_bias
result = self.base_layer(x)
if beft_bias is not None:
    result = (result + beft_bias).to(orig_dtype)

# merge / unmerge 状态机
if base_layer.bias is None:
    raise ValueError(f"Base layer has no bias, cannot merge bias adapter ...")
if safe_merge:
    output_bias = (base_layer.bias.data + beft_bias.squeeze()).clone()
    if not torch.isfinite(output_bias).all():
        raise ValueError(f"NaNs detected ...")
    base_layer.bias.data = output_bias.to(orig_dtype)
else:
    base_layer.bias.data.add_(beft_bias.squeeze()).to(orig_dtype)
# unmerge: base_layer.bias.data - beft_bias.squeeze()
```

三个关键口径：

1. **旁路而非原位**：beft_bias 是新增参数，原 weight/bias 在 forward 里原封不动——零初始化起点 forward ≡ base；BitFit 训的是**原** bias（`peft-bitfit-count` 的参数量账），两者实现分界。
2. **flatten（forward）与 squeeze（merge）**：官方 forward 用 `.flatten()`、merge 用 `.squeeze()`，对 \((1,out)\) 形状结果相同（均 \((out,)\)）——语义来源不同但数值不可分，勿变异钉。
3. **safe_merge 的检查对象是和**：`(base + beft).clone()` 后 `isfinite(...).all()`——不仅抓操作数 NaN，也抓两个有限数相加的溢出（1e308+1e308=inf）；非 safe 路径不拦截，NaN 照样写回。

**多 adapter 语义**：active_adapters 多于一个时 bias 先累加、base 输出只加一次；加性结构保证「逐个 unmerge」与「一次 unmerge 和」都恢复原 bias（浮点下 round trip 到 1e-12）。

## 易错点

1. **丢原 bias 或丢旁路**：forward = base + 旁路，两件缺一不可（m-fwd-nobase/m-fwd-nobeft 双钉）；权重忘了转置（m-fwd-wnot 钉）。
2. **广播轴向**：beft_bias 必须 reshape 成 \((out,)\) 沿最后一维广播； reshape 成列向量 \((out,1)\) 会广播到错误轴（m-fwd-col 钉）。
3. **merge 方向**：merge 是加、unmerge 是减，写反破坏 round trip（m-merge-sub/m-unmerge-add 双钉）。
4. **无 bias 护栏**：merge 时 base bias 为 None 必须抛 ValueError——静默补零向量会破坏「原 bias 存在」前提（m-merge-noguard 钉）；unmerge 无此护栏（官方只警告）。
5. **safe 拦截语义**：safe=False 时 NaN 合法返回（官方行为），只有 safe=True 抛（m-merge-nosafe 钉）；检查对象是和而非操作数——用「有限+有限溢出」fixture 钉（m-merge-safeop 实测 SURVIVED 后补 overflow 用例转 CAUGHT）。
6. **形状账**：beft_bias 是 \((1, out)\) 不是 \((out,1)\)（m-setup-shape 钉）；参数量恰为 out，无 in 维项（m-setup-count 钉）；零初始化是「训练起点 \(\Delta=0\)」的关键（m-setup-nozero 钉）。
7. **多 adapter 求和**：只取第一个 adapter 会丢后续 adapter 的贡献（m-sum-first 钉）；减法破坏累加（m-sum-sub 钉）。
8. **等价变异剔除记录**：`先 +base_bias 再 +beft` 与 `先 +beft 再 +base_bias` 是浮点加法交换律恒等（m-fwd-swap 实测 SURVIVED 剔除，勿再钉）；forward 的 flatten 与 merge 的 squeeze 在 \((1,out)\) 上同值同理。

## 对拍口径

锚点由沙箱 dump（torch float64，seed 42）：in=4/out=6、x (2,3,4)。互证三线——① forward ≡ x@Wᵀ + b + beft 显式展开；② merge 后「零旁路 + merged bias」的 forward ≡ 原 forward（bypass 收编等价）；③ merge→unmerge round trip ≡ 原 bias（单/多 adapter 双验）。safe_merge 用 NaN 操作数与 1e308 溢出两 fixture。16 变异 15 CAUGHT + 1 真等价剔除。

## 复杂度

forward 增量 \(O(out)\)（旁路加法），与 base 线性层 \(O(BN\cdot in\cdot out)\) 相比可忽略；merge/unmerge 各 \(O(out)\)；可训练参数 \(\sum_{\text{value 层}} out\)——0.01% 量级参数逼近全参微调（论文口径），是 peft 组参数最省的一题。
