# 数值稳定 Log-Softmax 详解

## 为什么要减最大值

softmax 里有 $e^{x}$：float32 在 $x\gtrsim 88$ 就 `inf`、$x\lesssim -87$ 就 `0`。
朴素 `log(softmax(x))` 会先得到一堆 `inf/inf=nan` 或 `log(0)=-inf`。利用平移不变性：
$$ \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}} $$
取 $m=\max_j x_j$ 后指数参数全 $\le 0$，最大项 $e^0=1$，既不上溢、分母又 $\ge 1$ 不下溢。

## 直接写 log 形式更稳

交叉熵 = `log_softmax` 接 `nll`。若分别算 `softmax` 再 `log`，小概率会先下溢成 0 再取
`log` 得 $-\infty$；`F.cross_entropy`/`log_softmax` 把两步合一，直接在 log 域输出，
是数值与速度的双重最优——这也是框架推荐用 `CrossEntropyLoss` 而非
`Softmax + NLL` 手动拼的原因。

## 梯度巧合

log-softmax 对 softmax 交叉熵的梯度极其干净：$\partial/\partial x_i = p_i - y_i$
（预测概率减 one-hot 标签）。这层「减 one-hot」正是分类损失反向传播的实现核心，
也解释了为何把 log 和 softmax 融合成一个算子既省显存又便于求导。

## 与其它题呼应

- 与「softmax」题共享减最大值技巧；log_softmax 只是把最后的除法写成 log 域减法。
- 与「attention-softmax-temp」相关：温度缩放后再 log_softmax 即带温度的对数概率。
- FlashAttention 的 online softmax（本合集）本质是「边分块边维护 running max/sum」
  的稳定 log-sum-exp，正是本题思想的流式版。
