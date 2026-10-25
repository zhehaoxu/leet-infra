# 多模态：CLIP 对称 InfoNCE 对比损失

## 题目描述

CLIP（Radford et al., arXiv:2103.00020）用**对称 InfoNCE** 对齐图像与文本两个编码器：把一个 batch 内 \(N\) 个图像特征与 \(N\) 个文本特征组成 \(N\times N\) 相似度矩阵，对角线是 \(N\) 个正样本对、其余是负样本。两个方向各做一次「把正确配对挑出来」的交叉熵，再取平均：

\[ \text{logits}_{i2t} = s\,(I T^\top),\quad \text{logits}_{t2i} = s\,(T I^\top),\quad y=\mathrm{arange}(N) \]
\[ \mathcal{L} = \tfrac12\bigl[\mathrm{CE}(\text{logits}_{i2t},y) + \mathrm{CE}(\text{logits}_{t2i},y)\bigr] \]

其中 \(s\) 为可学习温度缩放 `logit_scale`。

## 规则

实现 `solve(image_feats, text_feats, logit_scale)`。两特征均为已 L2 归一化的 float32、形状 \((N,D)\)；`logit_scale` 为标量。按上式返回 0 维 float32 损失标量。`CE` 用按行 softmax 的交叉熵（等价 `F.cross_entropy`），禁止直接调对比学习库。

## 示例

\(N=2,D=2\)：`image=[[1,0],[0.6,0.8]]`、`text=[[1,0],[0.2,0.98]]`、`logit_scale=4.0`：

```
logits_i2t = 4 * image@text.T ; labels=[0,1]
loss = 0.5*(CE(logits_i2t,[0,1]) + CE(logits_t2i,[0,1])) = 0.135395
```

## 约束

- CPU float32；输入特征视为已归一（本题不做归一化）。
- 对称 = i2t 与 t2i 两个方向、各自行上的交叉熵取平均。
- 温度 \(s\) 越大 → softmax 越尖锐（对负样本更惩罚）。
