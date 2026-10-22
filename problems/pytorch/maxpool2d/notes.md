# MaxPool2d 详解

## 定义

在每个滑动窗口内取最大值，输出 $(N,C,\Delta)\times$ 窗口数。作用：降采样 + 平移不变性 + 保留最强激活。

$$ \text{out}(n,c,i,j)=\max_{(p,q)\in\text{window}} x(n,c,i\cdot s+p,\,j\cdot s+q) $$

## 输出尺寸公式（务必记牢）

$$ H_{out}=\Big\lfloor\frac{H+2p-k}{s}\Big\rfloor+1 $$

`ceil_mode=True` 时把 $\lfloor\cdot\rfloor$ 换成 $\lceil\cdot\rceil$（多出一行/列，边缘用 $-\infty$ 填充再取 max，相当于允许不完整窗口）。stride 缺省 = kernel。

## padding 的坑

- max 的 padding 必须填 $-\infty$（或数据最小值），否则 0-padding 会**污染** max（全负输入里 0 反而成最大）。torch 内部对 maxpool 的 pad 就是 $-\infty$ 语义。
- 这也是 avgpool 与 maxpool 对 padding 处理不同的根源。

## 实现路线

- 简单正确：`F.max_pool2d(x, kernel_size, stride, padding)` 对拍。
- 手写：`unfold` 出窗口再 `.max(dim).values`；padding 用 `masked`/负无穷。

## 反向提示（理解用）

梯度只回传到窗口内取得最大值的那个位置（argmax 路由），其余为 0；这也是 maxpool 保留「最显著特征位置」的原因。avgpool 则把梯度均分给窗口所有位置。
