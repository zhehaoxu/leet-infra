# 按轴求和归约详解

## 「沿哪个轴」到底在压掉什么

`sum(dim=d)` 的直觉是：**把第 d 维抹平，其余维度保留**。二维里 `dim=0` 沿行方向竖着加、
得到每列的和（长度 = 列数）；`dim=1` 沿列方向横着加、得到每行的和（长度 = 行数）。
高维同理：结果形状 = 原形状去掉第 d 维。

$$ \text{out}_{\hat d} = \sum_{i} x_{[\hat d \leftarrow i]} $$

## keepdim 的意义

`keepdim=True` 保留被归约轴为长度 1，使结果仍能与原张量**广播回加**。典型用法：
softmax/layernorm 里 `x - x.mean(-1, keepdim=True)`——不 keepdim 就会因维度对不齐而广播错位。

## 与其它归约同构

把累加里的 `+` 换成 `max/min/·`、初值换成 $-\infty/+1$ 就得到 `amax/amin/prod`；
换成先求和再除以 $n$ 就是 `mean`。所以「一个轴 + 一个聚合函数 + 是否 keepdim」这三元组
足以描述所有归约算子，是理解 `torch.sum/mean/max/var` 的统一心智模型。

## 数值稳定性小记

沿大轴朴素累加会累积浮点误差；PyTorch 内部用成对求和（pairwise）降低误差。教学实现
用顺序累加即可，对拍时用 `allclose(atol=1e-5)` 容差而非精确相等。
