# Welford 在线方差：流式合并的三累量更新

## 题目描述

数据是一块一块到达的：dataloader 的 batch、流式训练的 shard、监控里的采样窗口。要在这条流上算出全局的 mean 与 var，教科书做法是攒 \(\sum x\) 与 \(\sum x^2\)，最后用 \(E[x^2]-E[x]^2\) 收尾——一旦数据带大偏置（比如全部是 \(10^9+k\)），两个几乎相等的大数相减会把这个量级的离散度整个抹掉。本题要求用 Welford 的单遍三累量 \((n, \text{mean}, M_2)\) 更新，逐元素推进，块与块之间不物化任何中间数据。返回 `(mean, var)` 两个 Python float。

## 规则

- 完整签名与参数含义只写在 starter 的 docstring 里，那是唯一判题口径。
- `chunks` 是 list/tuple，元素为 float32/float64 张量，按 row-major 展平成标量序列，允许空块；全程用 float64 累加。
- 每来一个标量 \(v\) 就做且只做这一步：

```
n += 1
d = v - mean
mean += d / n
M2 += d * (v - mean)
```

- 结尾 `unbiased=True` 给 \(M_2/(n-1)\)（样本方差），`unbiased=False` 给 \(M_2/n\)（总体方差）。
- 第 3、4 行的顺序不能颠倒：\(M_2\) 用的必须是**更新后**的 mean，写成 `d * d` 是最常见的错误。
- 不许先攒 sum 与 sum of squares 再用 \(E[x^2]-E[x]^2\) 收尾，也不许把数据物化后做两遍扫描。
- 空流（\(n=0\)）返回 `(nan, nan)`；\(n=1\) 时 mean 就是那个元素，`unbiased=True` 给 var=nan、`False` 给 0.0；流中出现 ±inf 或 NaN 也返回 `(nan, nan)`。
- `chunks` 不是 list/tuple、元素不是张量、dtype 非浮点、`unbiased` 不是 bool，都抛 ValueError。

## 示例

```
chunks = [1.0, 2.0] [] [3.0, 4.0]

unbiased=True  ->  mean = 2.5   var = 1.6666666666666667  (= 5/3)
unbiased=False ->  mean = 2.5   var = 1.25

offset = [1e9+1, 1e9+2, 1e9+3, 1e9+4]

welford           ->  mean = 1000000002.5   var = 1.6666666666666667
E[x^2] - E[x]^2   ->  var = 0.0
```

\(10^9+k\) 的离散度只有 1，而 \(\sum x^2\approx4\times10^{18}\) 的 ulp 已经是 1024，减掉 \((\sum x)^2/n\) 后剩下的 5 被舍成 0：求和式给出**恰好 0.0**，Welford 给出真值 \(5/3\)。空块插在中间不影响任何一步。

## 约束

- 单遍、\(O(1)\) 额外状态（三个标量）；块怎么切只影响末位，不影响 1e-9 容差下的结果
- oracle 用两遍法（先 `math.fsum` 求均值，再累加二阶中心矩），良态数据 1e-9 相对容差
- \(n\) 个连续整数的无偏样本方差是 \(n(n+1)/12\)：\(n=4\) 得 \(5/3\)，\(n=1000\) 得 83416.66666666667，可用来手算校验
