# 详解：binary_cross_entropy_with_logits（含 pos_weight 与 reduction）

## 出处

`F.binary_cross_entropy_with_logits` 的存在理由在 PyTorch 文档里写得非常直白：「数值上比先把 sigmoid 送进 `binary_cross_entropy` 更稳定」。多标签分类（COCO 标注、开放词表检测）、点击率模型、扩散模型的 ε-MSE 之外的二值掩码损失都会用到；`pos_weight` 是处理正负样本极度不均衡（正样本率 $10^{-3}$ 量级）的第一板斧，检测头的「正样本权重」就是它。

## 从概率形式推到稳定形式

设 $p=\sigma(x)$，逐元素损失（含 `pos_weight` $w$）为

$$\ell = w\,y\,-\log p + (1-y)\,-\log(1-p) = w\,y\log(1+e^{-x}) + (1-y)\log(1+e^{x})$$

关键的初等恒等式（分 $x>0$ 与 $x\le 0$ 两支直接验证）：

$$\log(1+e^{t}) = t^{+} + \underbrace{\log(1+e^{-|t|})}_{L}, \qquad t^{+}=\max(t,0)$$

$L$ 里的指数自变量永远是 $-|t|\le 0$，因此 $e^{-|t|}\in(0,1]$，`log1p` 精确；$|t|$ 很大时它下溢到 0，而 0 恰是 $\log(1+e^{-|t|})$ 的正确渐近值。代进去并用 $\max(-x,0)=\max(x,0)-x$：

$$\ell = w\,y\big(x^{+}-x\big) + (1-y)x^{+} + \big(w y + 1 - y\big)L = (1-y)x^{+} + w\,y(x^{+}-x) + \big(1+y(w-1)\big)L$$

$w=1$ 时退化为教科书写法 $\max(x,0)-xy+L$。参考答案就是这一行的逐字张量化，全程没有 `sigmoid`。

## reduction 的三种分母

| reduction | 分母 | 备注 |
| --- | --- | --- |
| `none` | — | 逐元素，已乘 `weight` |
| `sum` | 1 | |
| `mean` | `logits.numel()` | **不是** $\sum\text{weight}$ |

`mean` 这一条是真实的工程争论点：加权平均（除以 $\sum w$）在统计上更「像」均值，但 PyTorch 选择除以元素数，好让 `mean` 与 `sum` 之间只差一个常数因子、梯度尺度可控。判题用一组全 3 的 `weight` 直接验证：`mean(w=3) == 3·mean(w=None)`，而 $\big(\text{sum} \div \sum w\big)$ 会把它抵消成 1 倍——写错的人在这里必挂。`weight` 全 1 时必须与 `weight=None` 逐点相同。

## 核心与易错点

1. **`pos_weight` 只乘正类项**：$y=0$ 的项不含 $w$。等价校验式 `-(w*y*logsigmoid(x) + (1-y)*logsigmoid(-x)).sum()` 出现在判题里。
2. **`pos_weight` 是广播张量**：标量张量 `[2.0]` 与按类向量 `rand(5)` 都要支持，靠逐元素广播完成，不能 `.item()`。
3. **`weight` 乘在损失之后**：`(loss * weight)`，而不是乘 logits 或乘 targets。
4. **`reduction` 严格匹配字符串**：`"MEAN"` 大小写敏感，PyTorch 自己会抛 ValueError，判题要求同样抛出。
5. **float64 中间量**：`x=±1000` 时 `x⁺` 与 `L` 的相加在 float32 下会丢掉 $L$（$10^3$ 与 $10^{-434}$ 相加），虽然此时 $L$ 本就为 0，但 $x=\pm100$ 段的 $L\sim 3.7\times10^{-44}$ 仍在 float64 才有意义；判题容差 atol=1e-6 是按 float64 累加给的。

## 与邻近题的区别

- `bce-loss`：入参是**概率** $p\in[0,1]$，因此天然要做 clamp 且不稳；本题入参是 logits，禁止出现 sigmoid。
- `logsumexp`（order360）：同一族稳定化技巧（提最大值），但那题是归一化项、本题是 softplus 型。
- `focal-loss` / `dice-loss`：都在概率上做重加权或集合重叠，本题是逐样本二值对数损失。
- `cross-entropy-loss`：多类 $\log\sum e$ 口径；二项情形与本题等价，但 reduction 语义不同。

## 对拍口径与复杂度

- oracle 是 `F.binary_cross_entropy_with_logits`，`weight`（None/同形随机）、`pos_weight`（None/标量张量/按类向量）、`reduction`（三态）全组合笛卡儿积；另有大 logit 有限性探针、`mean` 分母探针、解析值 $\log 2$ 校验、4 个非法 `reduction`。
- 复杂度 $O(N)$，一次 `exp` + 一次 `log1p`，无归约以外的额外开销。
