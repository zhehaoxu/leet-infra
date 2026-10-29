# 多模态：Flamingo 媒体掩码与门控融合

## 任务

在含媒体标记的文本序列中，让每个 token 只读取它**最近一次遇到的媒体**，再通过一个可正可负的标量门，将媒体注意力结果加回文本特征。

本题有意抽取 OpenFlamingo 的 **attention 门控子层**，不是完整 `GatedCrossAttentionBlock`、Q-Former 或 OpenFlamingo 模型。不包含前馈层及其第二个门；无缓存，固定采用 `only_attend_immediate_media=True` 的语义。

参数、形状、数值范围和异常约定统一见 starter 的双语契约。以下只定义计算规则。

## 媒体可见性

`media_locations` 的 True 表示该文本位置出现了一个媒体标记，**不是 padding mask**。按文本顺序，第一个标记对应第一幅媒体，第二个对应第二幅，依此类推。

定义 `text_time = cumsum(media_locations)`。媒体按 1 到 M 编号；采用从零开始的数组下标时：

$$
\mathrm{visible}[b,t,m,k] = \bigl(\mathrm{text\_time}[b,t] = m+1\bigr).
$$

沿媒体轴、再沿每幅媒体的 latent 轴展开可见性与媒体 token，二者顺序必须一致。

- 标记位置**本身**即可读取它对应的媒体。
- 两个标记之间只能读取最近媒体，不能累计读取所有历史媒体。
- 第一处标记之前，或整行没有标记时，没有任何可见媒体。
- 未被标记引用的媒体不能影响结果；切换到下一幅后，上一幅不能泄漏。

## 注意力与门控

1. 仅对每个文本 token 的完整特征维做无 affine LayerNorm，使用总体方差：分母为 `sqrt(mean((x - mean(x))²) + eps)`。
2. 查询为 `LN(x) @ wq.T`。**media 不做 LayerNorm**；将其媒体和 latent 两维展开后，分别使用 `wk.T`、`wv.T` 投影。全部投影无 bias。
3. 按特征顺序切分 head，分数按 `sqrt(D/H)` 缩放。在可见媒体的 token 上执行稳定 softmax。其他位置的概率必须严格为零。
4. 没有可见媒体的行，其注意力概率和加权 value 结果必须**严格为零**，不能产生 NaN；不能把全 mask 行误当作均匀分布。
5. 按原 head 顺序合并加权结果并使用 `wo.T` 投影，最后计算：
   $$
   \mathrm{out}=x+\tanh(\mathrm{gate})\,\mathrm{projected}.
   $$

这是共享给所有 batch、token、head 和通道的 **tanh 标量门**，不是逐通道门，也不是 sigmoid 门。门为零时是恒等映射；正门加上对应方向的更新，负门反转该更新。改变 gate 不改变注意力概率或可见性。

返回残差结果、未乘门值的逐 head 注意力概率，以及展开后的布尔可见性。首图前输出严格等于原始文本特征。

## 样例 1：即时可见且只看最近媒体

全部浮点张量为 float64，四个权重均为二阶单位矩阵，`n_heads = 1`，使用默认 `eps`，`gate = log(3)/2 ≈ 0.5493061443340549`，因此 `tanh(gate) = 0.5`（数学值）。

```text
x = [[[1, -1], [0, 0], [0, 0], [0, 0]]]
media = [[[[2, 0], [0, 2]],
          [[4, 0], [0, 4]],
          [[9, 9], [9, 9]]]]
media_locations = [[false, true, false, true]]

out = [[[1, -1], [0.5, 0.5], [0.5, 0.5], [1, 1]]]
attention = [[[[0, 0, 0, 0, 0, 0],
               [0.5, 0.5, 0, 0, 0, 0],
               [0.5, 0.5, 0, 0, 0, 0],
               [0, 0, 0.5, 0.5, 0, 0]]]]
visible = [[[false, false, false, false, false, false],
            [true, true, false, false, false, false],
            [true, true, false, false, false, false],
            [false, false, true, true, false, false]]]
```

第一个 token 尚无媒体；第二、三个只看第一幅；第四个标记立即切换到第二幅。第三幅从未被引用。

## 样例 2：负标量门

同样使用 float64、二阶单位权重、一个 head 和默认 `eps`，但 `gate = -log(3)/2`。

```text
x = [[[2, -2], [0, 0]]]
media = [[[[2, 4]]]]
media_locations = [[false, true]]

out = [[[2, -2], [-1, -2]]]
attention = [[[[0], [1]]]]
visible = [[[false], [true]]]
```

负门只反转媒体更新，不反转概率，也不影响首图前的恒等映射。

以上输出均由参考实现实跑，并通过独立 oracle 校验；展示值保留六位小数。
