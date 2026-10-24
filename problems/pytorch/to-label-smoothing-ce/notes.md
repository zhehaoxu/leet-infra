# 解题说明 · Label Smoothing 交叉熵

## 从软标签交叉熵推两项式

目标分布 $q$ 是"正确类 $1-\alpha$、其余每类 $\alpha/V$"。代入 $\mathcal{L}=-\sum_v q_v\log p_v$：

$$-\Big[(1-\alpha)\log p_y + \tfrac{\alpha}{V}\sum_v \log p_v\Big] = (1-\alpha)(-\log p_y) + \alpha\cdot\tfrac{1}{V}\sum_v(-\log p_v)$$

注意平滑项对**全部 $V$ 类**求平均（正确类也被计入两次：一次在 NLL、一次在均值里），这正是题面用 `logp.mean(-1)`（而非"排除正确类"）的原因。

## 与 KL 的呼应

$\frac{1}{V}\sum_v(-\log p_v) = \mathrm{KL}(u\,\|\,p) + \mathcal{H}(u)$，其中 $u$ 为均匀分布、$\mathcal{H}(u)=\log V$ 是常数。所以 label smoothing 等价于"标准 CE + $\alpha\cdot\mathrm{KL}(\text{uniform}\|p)$"，把预测往均匀推、惩罚过度自信——与 operators 里的 `kl_div` 同源。

## 数值稳定

务必用 `F.log_softmax`（内部先减行最大值再 logsumexp）。若写 `torch.log(F.softmax(x))`，大 logit 会让 softmax 上溢/下溢成 0，log 得 $-\infty$。测试用 `1000.0` 级 logit 与"全相等 logit→loss=ln V"验证稳定性与正确性。

## α=0 的自检

$\alpha=0$ 时必须与 `F.cross_entropy(reduction='mean')` 逐位相等——这是最有力的正确性锚点，测试专门断言。
