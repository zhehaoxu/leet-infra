# 检测框 IoU 变体损失：GIoU / DIoU / CIoU

## 题目描述

\(1-\mathrm{IoU}\) 是最朴素的目标框回归损失，但它在两框**不相交**时梯度为 0（IoU 恒 0），也区分不出"框偏了"与"框瘦了"。三个变体各自补一块：GIoU（Rezatofighi et al., CVPR 2019, arXiv:1902.09630）减去**外接闭包占比** \((C\text{面积}-\text{union})/C\text{面积}\)，使不相交时仍有方向性；DIoU（Zheng et al., AAAI 2020, arXiv:1911.08287）把该项换成**中心距平方比** \(\rho^2/c^2\)，收敛更快；CIoU 在 DIoU 上再减 \(\alpha v\)，用宽高比一致性把长宽失衡也计进损失。本题一次算三件并返回 \((1-\mathrm{GIoU},\,1-\mathrm{DIoU},\,1-\mathrm{CIoU})\)。CPU、全程确定性、无随机数调用、只算前向（给定上游梯度求局部梯度不在本题范围）。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。框口径统一 \([x_1,y_1,x_2,y_2]\)，pred/target 同形状 (N,4) 逐行配对；输入保证是合法框（\(x_2>x_1\)、\(y_2>y_1\)），**不需要**你校验；N=0 允许；reduction 只接受 `"mean"`（默认）与 `"none"`，其他取值抛 ValueError。
- **基础 IoU 与 `iou-compute` 完全同口径**：交集宽高各自 `clamp_min(0)`，\(\text{inter}=\text{iw}\cdot\text{ih}\)，\(\text{union}=A_p+A_t-\text{inter}\)，\(\text{IoU}=\text{inter}/\text{union}\)；\(\text{union}=0\)（两框都是空框）时 IoU 记 0。
- **外接闭包 C**：\(C_{x_1}=\min(a_{x_1},b_{x_1})\)、\(C_{y_1}=\min(a_{y_1},b_{y_1})\)、\(C_{x_2}=\max(a_{x_2},b_{x_2})\)、\(C_{y_2}=\max(a_{y_2},b_{y_2})\)，宽高 `clamp_min(0)`；\(C\text{面积}=w_C h_C\)，\(c^2=w_C^2+h_C^2\)（对角线平方）。
- **三个式子**（惩罚项在各自分母为 0 的退化情形一律记 0，保证输出有限）：

  \[\text{GIoU}=\text{IoU}-\frac{C\text{面积}-\text{union}}{C\text{面积}},\qquad
    \text{DIoU}=\text{IoU}-\frac{\rho^2}{c^2},\qquad
    \text{CIoU}=\text{DIoU}-\alpha v\]

  其中 \(\rho^2\) 是两框**中心点距离平方**，\(v=\frac{4}{\pi^2}\big(\arctan\frac{w_t}{h_t}-\arctan\frac{w_p}{h_p}\big)^2\)，\(\alpha=\dfrac{v}{(1-\text{IoU})+v}\)。宽高比除法参与前，\(w,h\) 各自 `clamp_min(1e-7)`；\((1-\text{IoU})+v=0\)（两框完全重合）时 \(\alpha\) 记 0。
- 损失都是 **\(1-\) 相似度**：返回 3 元 tuple，顺序固定 `(loss_giou, loss_diou, loss_ciou)`。`"none"` 给三个 (N,) 张量，`"mean"` 给三个 0 维标量；N=0 且 `"mean"` 时按空集约定返回三个 0。

## 示例

pred \(=[[0,0,2,2],[0,0,1,2]]\)，target \(=[[1,1,3,3],[0.5,0.5,2.5,2.0]]\)。

**pair 0**（两个 \(2\times2\) 正方形对角错开 1 格）：\(\text{iw}=\text{ih}=1\)、inter=1、\(A_p=A_t=4\)、union=7 → IoU \(=1/7=0.142857\)；闭包 \(3\times3\) → \(C\text{面积}=9\)、\(c^2=18\)；中心 \((1,1)\) 与 \((2,2)\) → \(\rho^2=2\)。两框皆正方形 → \(v=0\)、\(\alpha=0\)，CIoU 退化成 DIoU。逐对损失（EXAMPLE_NONE_*）：

| loss_giou | loss_diou | loss_ciou |
|---|---|---|
| 1.0793650150 | 0.9682539701 | 0.9682539701 |

GIoU \(=1/7-2/9=-0.079365\) 落在 \([-1,1]\) 之外的一侧，因此损失 **>1**——这正是不相交（此处是角部重叠很少）时 GIoU 比 \(1-\)IoU 更"惩罚"的体现。

**pair 1**（\(1\times2\) 竖框 vs \(2\times1.5\) 扁框）：inter \(=0.5\times1.5=0.75\)、union \(=2+3-0.75=4.25\) → IoU \(=0.176471\)；闭包 \(2.5\times2\) → \(C\text{面积}=5\)、\(c^2=10.25\)；中心 \((0.5,1)\) 与 \((1.5,1.25)\) → \(\rho^2=1.0625\)；\(v=\frac4{\pi^2}(\arctan\frac2{1.5}-\arctan\frac12)^2=0.087123\)、\(\alpha=0.087123/0.910652=0.095670\)。逐对损失：

| loss_giou | loss_diou | loss_ciou |
|---|---|---|
| 0.9735293984 | 0.9271879196 | 0.9355232120 |

**默认 `mean`**（EXAMPLE_*）：loss_giou = 1.0264471769、loss_diou = 0.9477209449、loss_ciou = 0.9518885612，恰为上面两行各自的两两平均。

## 约束

- pred/target 形状同为 (N,4)，N 可为 0；框口径 \([x_1,y_1,x_2,y_2]\)；不含 batch/类别维，不做坐标缩放或 cxcywh 转换
- reduction ∈ {`"mean"`, `"none"`}；`"sum"` 等一律 ValueError；三值顺序固定，`mean` 返回 0 维标量张量、`none` 返回 (N,)
- 数值约定：所有除法先判分母（union、C面积、\(c^2\)、\((1-\text{IoU})+v\)）为 0 则该惩罚项取 0，不允许出现 NaN/Inf；aspect 除法前 \(w,h\) clamp 到 \(10^{-7}\)
- 对拍容差 float32 \(10^{-5}\)（atol 与 rtol 同）；判题另有不等式探针：`pred==target` 三损失恒 0、\( \text{loss}_{ciou}\ge \text{loss}_{diou}\ge 1-\text{IoU}\)、不相交时 loss_giou>1、全正方形输入时 loss_ciou 与 loss_diou 逐对相等
