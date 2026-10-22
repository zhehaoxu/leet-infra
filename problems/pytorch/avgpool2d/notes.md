# AvgPool2d 详解

## 定义

在每个窗口内求平均。输出尺寸公式与 maxpool 相同：

$$ H_{out}=\Big\lfloor\frac{H+2p-k}{s}\Big\rfloor+1 $$

## 核心坑：`count_include_pad`

这是 AvgPool 独有的语义分叉，决定分母：

- `True`（torch 默认）：**分母恒为窗口面积 $k_h k_w$**，padding 的 0 计入分子也计入分母 → 边缘输出被「稀释」偏小。
- `False`：**只除以有效（非 pad）像素数**，边缘窗口分母更小 → 不因补零而降权。

两种都对，但数值不同；判题口径要对齐。实现技巧：对 0-pad 的**全 1 掩码**走同样的 unfold 求和，得到每窗口的有效计数作分母。

## 实现路线

`F.avg_pool2d(x, kernel_size, stride, padding, count_include_pad)`。手写用 `unfold(2,kh,sh).unfold(3,kw,sw).sum((4,5))` 再除分母。padding 这里填 0 即可（与 maxpool 的 $-\infty$ 相反，因为 0 对求和是中性、只是要处理计数）。

## 与全局平均池化

把窗口设成整幅特征图即 GAP（见「全局池化」关），是 ResNet 等替换全连接的关键。
