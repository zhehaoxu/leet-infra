# 多模态：带掩码的 Perceiver Latent 压缩

## 任务

用一组 latent 查询视觉 token，完成**一次 Perceiver 注意力残差更新**。关键之处是：key/value 不仅包含视觉 token，也包含 latent 自身，因此即使所有视觉 token 都不可见，注意力仍然有合法来源。

本题有意抽取 OpenFlamingo 的 Perceiver 子层，**不是完整 PerceiverResampler，也不是 Q-Former 或完整 OpenFlamingo 模型**。`visual_mask` 是本题新增的扩展，原始 OpenFlamingo Perceiver 接口没有这个参数；本题不包含前馈层、末尾 LayerNorm、时间或帧嵌入。

参数、形状、数值范围和异常约定统一见 starter 的双语契约。以下只定义计算规则。

## 计算规则

1. 对每个 token 的特征维单独做无 affine LayerNorm：
   $$
   \operatorname{LN}(z)=\frac{z-\mu}{\sqrt{\frac{1}{D}\sum_i(z_i-\mu)^2+\varepsilon}},
   \qquad \mu=\frac{1}{D}\sum_i z_i.
   $$
   视觉和 latent 分别归一化；不能混合 batch、token 或 head 维计算统计量。
2. 查询为 `LN(latents) @ wq.T`。将 `LN(visual)` 与 `LN(latents)` **按此顺序沿 token 维拼接**，再分别使用 `wk.T`、`wv.T` 投影，得到 key 和 value。所有投影均无 bias。
3. 将特征维顺序切成 `H` 个 head，每个 head 的宽度为 `d_h = D/H`。分数为 `Q_h @ K_h.T / sqrt(d_h)`。
4. key 的可见性为 `[visual_mask, K 个 True]`，广播到所有 head 和查询。沿 key 维做稳定 softmax；不可见位置的概率必须严格为零。latent key 永远可见，不存在全 mask 的查询行。
5. 每个 head 对 value 加权求和，按原 head 顺序合并特征，使用 `wo.T` 投影，再加上**未经归一化的原始 latents**。

返回残差更新结果及每个 head 的注意力概率。概率最后一维的顺序是所有视觉 token，随后所有 latent；它不是分数，也不是 head 平均值。

## 样例 1：视觉掩码与 latent key

全部浮点张量为 float64，四个权重均为二阶单位矩阵，`n_heads = 1`，使用默认 `eps`。

```text
visual = [[[1, -1], [-1, 1]]]
latents = [[[0, 0], [1, -1]]]
visual_mask = [[true, false]]

output = [[[0.666663, -0.666663], [1.891611, -1.891611]]]
attention = [[[[0.333333, 0, 0.333333, 0.333333],
               [0.445808, 0, 0.108385, 0.445808]]]]
```

第二个视觉 token 不可见，但两个 latent key 均参与归一化。第一条查询为零，所以三个可见 key 等概率。

## 样例 2：没有视觉 token

全部浮点张量为 float64，`n_heads = 1`，使用默认 `eps`。

```text
visual = 空张量，形状 (1, 0, 1)
visual_mask = 空 bool 张量，形状 (1, 0)
latents = [[[3]]]
wq = wk = wv = wo = [[1]]

output = [[[3]]]
attention = [[[[1]]]]
```

单特征 LayerNorm 的结果为零，但唯一的 latent key 仍有概率 1，残差保留原值。

以上输出均由参考实现实跑，并通过独立 oracle 校验；展示值保留六位小数。
