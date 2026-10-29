# logsumexp（减最大值稳定式与 ±inf 退化段）

## 题目描述

$\log\sum e^{x}$ 是 softmax、交叉熵、变分下界、序列似然里无处不在的一处归一化项。朴素写法 `log(sum(exp(x)))` 有两处必挂：$x\sim 10^3$ 时 `exp` 上溢成 `inf`、$x\sim -10^3$ 时全部下溢成 0 再 `log` 得 `−inf`。标准修法是提出最大值，把指数的自变量压到 $(-\infty, 0]$：

$$\log\sum_i e^{x_i} = m + \log\sum_i e^{\,x_i - m}, \qquad m = \max_i x_i$$

第二道坎是退化段：掩码把一整行 logits 全置 $-\infty$ 时 $m=-\infty$，$x_i - m$ 变成 $-\infty-(-\infty)=\mathrm{NaN}$ 并污染整段；段内只要有 $+\infty$ 结果就应是 $+\infty$。这两支都必须显式取值，不能指望稳定式自动扛住。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `dim=None` 是本题在 torch 之外的扩展口径（`torch.logsumexp` 的 `dim` 是必填的）：整体归约返回 **0 维**张量；此时 `keepdim=True` 要求**所有轴都保留为长度 1**。
- `dim` 为 int 或 int 元组时与 `torch.logsumexp` 完全一致，负数按 `x.dim()` 折算；越界、重复轴、空元组抛 ValueError。
- 退化口径：整段全 $-\infty$ → $-\infty$（**不是 NaN**）；段内含 $+\infty$ → $+\infty$；段内含 NaN → 传播 NaN。
- 禁止用 `x.exp().sum().log()` 这种朴素式；内部 float64 计算，返回 float32。

## 示例

$X$ 为 $2\times 3$（数值取自 test_cases.py 回填的 EXAMPLE_* 常量）：

```
X = [[   0,    1,    2],
     [-100,  -99,  -98]]

dim=1 -> [  2.4076058864593506, -97.59239196777344]
dim=0 -> [  0.0,  1.0,  2.0]
(0,1) ->  2.4076058864593506
```

第一行 $m=2$，$2 + \log(e^{-2}+e^{-1}+1) = 2 + \log(1.50827) = 2.4076$；第二行 $m=-98$，整体平移不改变「$m$ 加上一个 $[0,\log 3]$ 之间的修正量」这一结构，因此结果是 $-97.5924$ 而不是 `−inf`。

## 约束

- 输入 float32，可含 $-\infty$ / $+\infty$ / NaN；与 `torch.logsumexp` 对拍 atol=1e-5、rtol=1e-5
- 判题含 $10^3$ 与 $-10^3$ 两段溢出探针（并先断言朴素式确实失效）、`log_softmax(x) = x − logsumexp(x)` 的一致性交叉验证、非法 `dim` 探针
