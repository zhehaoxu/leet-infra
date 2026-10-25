# 详解：从 logits 计算困惑度 PPL

## 出处
- Mikolov et al., *The Recurrent Neural Language Model*（2012）：PPL \(=\exp\big(\frac1N\sum_{i=1}^{N}\text{NLL}_i\big)\)，N 为参与评测的 token 数。
- HuggingFace `evaluate` / `examples/pytorch/language-modeling` 的 `perplexity`：先算平均 NLL（`CrossEntropyLoss(ignore_index=…)`）再取指数；这正是本题口径。
- 与 `cross-entropy-loss`/`to-label-smoothing-ce` 的区别：那些题算的是**训练用损失张量**，本题要的是**评测指标**——右移对齐、掩码计数、哨兵约定和「micro 平均」都在本题首次显式化。

## 核心与易错点
1. **右移而不是不移/移反**：causal LM 里 `logits[b, i]` 是看完 `token[0..i]` 后对 `token[i+1]` 的分布。所以配对是
   \[ \big(\text{logits}[b, i],\ \text{labels}[b, i+1]\big),\quad i = 0 \dots L-2. \]
   有效位总数最多 \(B(L-1)\)。丢掉的是**最后一个位置的 logits**和**第 0 个标签**，两者方向相反，写反就全错。
2. **gather 的是 log-softmax，不是 softmax 再 log**：\(-\log\mathrm{softmax}(z)_y = \log\sum_v e^{z_v} - z_y\)。实现上先减行最大值 \(m\)：\(\text{nll}= \big(m+\log\sum_v e^{z_v-m}\big)-z_y\)。直接 `log(softmax)` 在低温度/大 logit 时会因下溢得到 \(\log 0\)。
3. **ignore_index 只作用在 shift 之后的标签上**：\(\text{labels}[b,0]\) 无论是不是 \(-100\) 都不参与；prompt 段的掩码效果要通过 `data-collator` / `sft-assistant-mask` 写进 labels。别在 logits 那侧造第二个 mask，两个 mask 与一次 shift 叠加会重复扣位。
4. **-100 不能拿去 gather**：`gather` 的索引必须非负且 \(<V\)；正确做法是先算 mask，再对 `labels.clamp(min=0)` 取值，最后用 mask 过滤。若把 \(-100\) 直接喂进 `F.cross_entropy(..., ignore_index=-100)` 是安全的（PyTorch 内部会跳过），但手写 `log_softmax(...)[..., labels]` 时越界索引会静默拿到错误元素。
5. **分母是有效位数，不是 B·L**：pad/prompt 不计数。用 `nll.mean()` 而非 `nll[valid].mean()` 会把哨兵位算成 0 损失，从而**低估**困惑度（PPL 偏低）。
6. **micro vs macro**：本题把所有有效 token 放进一个池子求平均。若先对每条样本求均值再对样本求均值（macro），短句与长句权重相同，数字会漂——这个不对称在 `eval-token-accuracy` 里作为主考点，PPL 上同理。
7. **哨兵约定**：\(n=0\) 时 PPL 无定义，取 \(-1.0\) 而不是 0/NaN/inf，理由是真实 PPL \(\ge 1\)，\(-1\) 与值域不重叠且可精确比较（NaN 与 `allclose` 不友好，inf 容易掩盖实现错误）。

## 数学推导（指标含义）
\[ \text{PPL} = \exp\Big(-\frac1n\sum_{i\in\mathcal V}\log p_\theta(x_{i+1}\mid x_{1..i})\Big) = \Big(\prod_{i\in\mathcal V} p_\theta(x_{i+1}\mid x_{1..i})\Big)^{-1/n} \]
即"逐位预测概率的几何平均的倒数"。几何平均（而非算术平均）正是取指数/对数的原因：单个极低概率位置不会被稀释。
边界值：完全确定时每个 \(p=1\)，\(\text{PPL}=1\)；词表均匀时每个 \(p=1/V\)，\(\text{PPL}=V\)。这两条是本题的两枚探针。

## 对拍口径
参考解向量化：切片 shift → `log_softmax` → `gather` → 布尔索引取有效位 → `exp(mean)`。判题 `_oracle` 用三层 Python 循环逐位手算 \(\big(m+\log\sum e^{z-m}\big)-z_y\) 并累加计数（结构完全不同、数值等价），`allclose(atol=1e-5, rtol=1e-5)`。另设四枚退化断言：全 `ignore_index` → \(-1.0\)、\(L=1\) → \(-1.0\)、均匀 logits → \(V\)、one-hot 强预测 → \(\approx 1\)；再加「只有 1 个有效位时等于该位 \(1/p\)」与「移位方向写错会命中 \(-1.0\) 哨兵」的定向探针。

## 复杂度
\((B, L-1)\) 个位置，每个位置一次 \(O(V)\) 的 log-sum-exp：\(O(BLV)\) 时间、\(O(BL)\) 额外空间（若先物化 log-softmax 则是 \(O(BLV)\)）。

## 实践提醒
评测时把 logits 转 float32（哪怕模型是 bf16），否则 `log_softmax` 的舍入误差在长序列上会放大 PPL；对比不同 checkpoint 时务必固定同一份 labels/mask 口径，否则 \(n\) 不同、PPL 不可比。
