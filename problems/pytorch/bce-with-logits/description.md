# binary_cross_entropy_with_logits（含 pos_weight 与 reduction）

## 题目描述

多标签分类、OCR/检测头的置信度损失、推荐里的点击模型都用二元交叉熵，而工程上**绝不能**先过一层 sigmoid 再取对数：logits 到 $\pm10^3$ 量级时 `sigmoid` 直接饱和成 0 或 1，`log 0` 给出 `−inf`，训练一遇大 logit 就 NaN。PyTorch 的 `binary_cross_entropy_with_logits` 把两者合并成代数上等价、数值上稳定的形式：

$$\ell_i = -\Big[y_i\log\sigma(x_i) + (1-y_i)\log\big(1-\sigma(x_i)\big)\Big]$$

利用 $\log\sigma(x) = -\log(1+e^{-x})$、$\log(1-\sigma(x)) = -\log(1+e^{x})$ 以及恒等式 $\log(1+e^{t}) = t^{+} + \log(1+e^{-|t|})$，可写成不会出现上溢的形式（记 $L=\log(1+e^{-|x|})$，$w$ 为 `pos_weight`，缺省 1）：

$$\ell_i = (1-y_i)\max(x_i,0) + w_i\,y_i\big(\max(x_i,0) - x_i\big) + \big(1 + y_i(w_i - 1)\big)L$$

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 禁止 `sigmoid` 后再 `log`；判题用 $x\in\{\pm10^3,\pm10^2,0\}$ 探针断言输出全部有限。
- `reduction`：`"none"` 返回逐元素损失（**已乘 `weight`**，形状同 logits）；`"sum"` 求和；`"mean"` 是**和 ÷ logits.numel()**，分母不是 $\sum\text{weight}$。
- `pos_weight` 只作用在正类项上（$y=1$ 那一项乘以 $w$），可与 logits 广播；`weight` 是逐元素乘在整个损失上的样本权重。
- `reduction` 取其他值（含 `"MEAN"`、`None`、空串）抛 ValueError。

## 示例

$x=[-2,-0.5,0,1,3]$、$y=[1,0,1,0,1]$（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
loss(none) = [2.1269280910491943, 0.4740769863128662, 0.6931471824645996,
              1.31326162815094,   0.04858735203742981]
mean       = 0.9312002658843994
sum, pos_weight=2 -> 7.524663925170898
```

第一项 $x=-2, y=1$：$-\log\sigma(-2)=\log(1+e^{2})=2.1269$；第三项 $x=0,y=1$：$-\log 0.5=\log 2=0.6931$。`mean` 的分母是 5 个元素；`pos_weight=2` 把三个 $y=1$ 的项翻倍后求和得 $7.5247$。

## 约束

- 输入 float32，内部 float64 计算，返回 float32；与 `F.binary_cross_entropy_with_logits` 对拍 atol=1e-6、rtol=1e-5
- `weight` / `pos_weight` 的 None、标量张量、按类向量三种形态都要覆盖；`weight` 全 1 时必须与 `weight=None` 逐点相同
- 与 `bce-loss`（输入概率）的分工：那题考的是已经过 sigmoid 的概率入参，本题禁止出现 sigmoid。
