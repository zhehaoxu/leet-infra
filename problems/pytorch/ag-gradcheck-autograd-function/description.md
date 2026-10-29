# 梯度校验：autograd.Function 解析梯度与中心差分的对拍

## 题目描述

写完自定义 `autograd.Function` 之后，唯一可信的验证方法是把它的解析梯度与**有限差分**对拍。这里有三个反复出现的错误：用 `torch.autograd.grad(forward(...))` 去取「解析梯度」（那是在校验 PyTorch 自己的反向，自定义 backward 根本没被执行）、只检查第一个输入或前几个元素、以及把步长取成 `eps` 本身（截断误差与舍入误差同时爆炸）。本题给你 `forward, backward, inputs`，要求返回逐输入的**最大相对误差列表**。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 解析梯度**必须**经过 `autograd.Function`：`forward` 里算前向并保存入参、`backward` 里调用给定的 `backward`，再用 `torch.autograd.grad` 取梯度。
- 数值梯度对**每个输入的每个元素**做中心差分 \((L(x+\delta)-L(x-\delta))/(2\delta)\)，步长 \(\delta_i=\sqrt{eps}\,(1+|x_i|)\)（`eps` 为入参，默认 float64 机器精度）。
- 相对误差 \(rel=|a-n|/\max(|a|,|n|,10^{-8})\)；返回 `list[float]`，第 \(k\) 项是 `inputs[k]` 全部元素的最大值。
- `inputs` 含非 float64 张量、`forward` 返回的不是 0 维张量、`backward` 返回个数与 `inputs` 不符，一律抛 ValueError。

## 示例

\(L=\sum x^2\) 的解析梯度是 \(2x\)，数值梯度对二次函数**精确**（\((x+\delta)^2-(x-\delta)^2=4x\delta\)），误差只剩舍入量级：

```
inputs = [ [[1.0, -2.0]] ]
max_rel_error = [ ~1e-16 ]        # asserted < 1e-10
```

## 约束

- 只支持 float64 输入（步长与误差阈值都按 float64 设计），`inputs` 可为 1~3 个张量、形状任意
- 判题同时放「正确 backward ⇒ 误差极小」与「故意写错的 backward ⇒ 误差必须超过阈值」两类用例，并用三输入只错第三个来定位漏检
