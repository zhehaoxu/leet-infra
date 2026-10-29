# 交叉熵反向：带 label smoothing 与 ignore_index 的 logits 梯度

## 题目描述

训练语言模型时，`F.cross_entropy` 是每一步都要穿过损失函数的算子；它的反向只有一行 \(p - y\)，但工程实现里那一行要同时处理三件麻烦事：label smoothing 把 one-hot 摊成 \((1-\varepsilon)\delta + \varepsilon/C\)、`ignore_index` 让部分行整行梯度为 0 且不进 `mean` 的分母、`reduction='sum'` 又要求完全不做除法。本题给你 `logits (N, C)` 与 `target (N,)`，要求一次算出与 logits 同形的梯度，把这三条口径同时做对。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 单个有效行的梯度为 \(dL/dz_k = p_k - (1-\varepsilon)\,onehot_y[k] - \varepsilon/C\)，其中 \(p=\text{softmax}(logits)\)；\(\varepsilon=0\) 时退化为 \(p-y\)。
- 被忽略（等于 `ignore_index`）的行整行为 0，并且 `mean` 的分母是**有效行数**，不是 N。
- `label_smoothing` 必须落在 \([0,1)\)；`reduction` 只接受 `'mean'`/`'sum'`；非法入参一律抛 ValueError。
- 数值上先做 logits 的最大值平移再取 softmax，禁止先算 log 再乘 one-hot 导致被忽略行出现 `nan`。

## 示例

4 类、2 行、`reduction='mean'`、\(\varepsilon=0\)，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
logits = [[0, 0, 0, 0],      target = [1, 0]
          [0, 0, 0, 0]]      valid rows = 2
p      = [0.25, 0.25, 0.25, 0.25]
dL/dz  = [[ 0.125, -0.375,  0.125,  0.125],
          [-0.375,  0.125,  0.125,  0.125]]
```

每行是 \((0.25-\text{onehot})/2\)：命中类 \(0.25-1=-0.75\) 除以 2 得 \(-0.375\)。

## 约束

- logits 为 float32 或 float64；与 `torch.autograd.grad(F.cross_entropy(...))` 对拍，float64 atol=1e-9
- 内部 float64 累计，返回 dtype 与 logits 一致；`target` 全为 `ignore_index` 时返回全 0 且不抛除零
