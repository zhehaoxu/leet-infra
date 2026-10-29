# conv2d 反向：dx 的翻转核与 dw 的相关

## 题目描述

卷积反向是两段互为对偶的运算：\(dx\) 是「把上游梯度摊回输入」，等价于用**旋转 180° 的核**做一次全卷积；\(dw\) 是「把输入与上游梯度按窗口配对求和」，形式上就是普通的相关。二者共用同一套 `stride`/`padding` 索引，却各自有独立的边界处理——\(dx\) 要丢弃越出输入范围的贡献，\(dw\) 要把 batch 与空间轴统统折叠掉。本题给你 `dY, x, weight`，返回 `(dx, dw)`。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- 前向 \(Y[b,co,u,v]=\sum_{ci,k,l}x[b,ci,u\cdot s+k-p,\,v\cdot s+l-p]\cdot weight[co,ci,k,l]\)。
- \(dw[co,ci,k,l]=\sum_b\sum_{u,v}dY[b,co,u,v]\cdot x[\dots]\)，只对落在 \(x\) 范围内的 \((u,v)\) 求和；形状恒等于 `weight`。
- \(dx\) 用翻转核与 `dY` 做相关，padding 造成的虚拟位置**不**写进 `dx`，`dx` 形状恒等于 `x`。
- 仅支持 `groups=1`；`stride<1`、`padding<0`、`Cin` 与核不符、\(H_{out}\ne(H_{in}+2p-k_h)/s+1\) 一律抛 ValueError。

## 示例

\(3\times3\) 输入、\(2\times2\) 全 1 核、`stride=1`、`padding=0`，数值取自 test_cases.py 的 EXAMPLE_* 常量：

```
x     = [[1, 2, 3],     weight = [[1, 1],     dY = [[1, 0],
       [4, 5, 6],                [1, 1]]           [0, 1]]
      [7, 8, 9]]
dx    = [[1, 1, 0],     dw = [[6,  8],
        [1, 2, 1],            [12, 14]]
        [0, 1, 1]]
```

\(dw[1,1]=x_{1,1}+x_{2,2}=5+9=14\)；\(dx_{1,1}\) 被四个输出窗口同时覆盖，其中两个 \(dY\) 非零 ⇒ 2。

## 约束

- 内部 float64 累计，`dx`/`dw` 的 dtype 分别与 `x`/`weight` 一致
- 与 `F.conv2d` + `torch.autograd.grad` 对拍（`stride`、`padding` 同步传入），float64 atol=1e-9
