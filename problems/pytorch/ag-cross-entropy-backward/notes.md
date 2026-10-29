# 详解：交叉熵反向的三条工程口径

## 出处

`F.cross_entropy` 的反向是 LLM 预训练与 SFT 里被问得最多的一题：它同时是「softmax 反向 + 取 one-hot」的合并化简，也是 label smoothing、`ignore_index`（padding 不参与损失）与 `reduction` 三种约定交汇的地方。加分项通常出在分母：很多人会把 `mean` 除以 N 而不是有效行数。

## 从 softmax 反向推到 p − y

交叉熵 \(L=-\sum_k y_k\log p_k\)，把 softmax 反向的 \(dx_i=y_i(grad_i-\sum_k y_k grad_k)\) 与 \(grad_k=-y_k/p_k\) 复合：

$$dx_i = p_i\Big(-\frac{y_i}{p_i}-\sum_k y_k\Big(-\frac{y_k}{p_k}\Big)\Big) = -y_i + p_i\sum_k\frac{y_k^2}{p_k}$$

\(y\) 是 one-hot 时 \(\sum_k y_k^2/p_k=\sum_k y_k/p_k=1/p_{argmax}\)，且 \(p_{argmax}\) 抵消，得到 \(p_i-y_i\)。也就是说 **\(p-y\) 只对 one-hot 成立**；一旦 label smoothing（\(y\) 不再 0/1），必须回到原始形式 \(p_k-(1-\varepsilon)onehot-\varepsilon/C\)，而不能写成 \(p-\tilde y\) 后再除 C——这是两条不同的 \(\varepsilon/C\) 分摊口径。

## 核心与易错点

- **忽略行的分母**：PyTorch 的 `mean` 是 `sum / max(count_nonzero(ignore), 1)`，全忽略时不报错而给 0。
- **\(\varepsilon/C\) 分摊到全部 C 列**，包含真类别那一列，所以真类别的系数是 \(-(1-\varepsilon)-\varepsilon/C\)。
- **数值稳定**：先减行最大值再 exp；被忽略行也要正常算 softmax（梯度仍要置 0），不要提前 mask logits 造成 `-inf - -inf`。
- **int64 target**：用 scatter 写 one-hot 比逐行 `arange` 索引快，且 `ignore_index=-100` 直接作下标会越界，必须先建有效位掩码。

## 与邻近题的区别

`ag-softmax-backward` 拿到的已经是归一化后的输出，只需一次归约；本题从 logits 出发，还要处理 target 与三种归约约定。`to-label-smoothing-ce` 只求平滑后的**损失值**（正向），这里求的是**梯度**，而且平滑与忽略是耦合的：\(\varepsilon/C\) 只加在有效行上。

## 对拍口径与复杂度

判题用 `torch.autograd.grad(F.cross_entropy(logits, target, label_smoothing=..., ignore_index=..., reduction=...), logits)` 作 oracle，float64 atol=1e-9；示例是 logits 全 0 的均匀分布，可手算逐位验证。复杂度 \(O(NC)\) 时间、\(O(NC)\) 额外空间（复用 softmax 缓冲可降到 \(O(NC)\) 常数次）。
