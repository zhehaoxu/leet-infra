# 详解：检测框 IoU 变体损失

## 出处

- GIoU：Rezatofighi et al., *Generalized Intersection over Union*（CVPR 2019, arXiv:1902.09630）。原文用外接矩形的"空洞占比"补上 IoU 在不相交时梯度为 0 的缺陷，并证明 GIoU ∈ \([-1,1]\)。
- DIoU / CIoU：Zheng et al., *Distance-IoU Loss*（AAAI 2020, arXiv:1911.08287）。DIoU 把重叠驱动改为"重叠 + 中心距"，CIoU 再加宽高比项 \(v\) 与权衡系数 \(\alpha\)；题面采用论文式（\(\alpha=v/((1-\text{IoU})+v)\)），不是后续 NWD/EIoU 分支。
- 基础 IoU 口径继承本仓库 `iou-compute`（交集宽高 clamp_min(0)、union 为 0 时 IoU 记 0），三损失统一为 \(1-\)相似度。
- 本题只算前向：「给定上游梯度求 \(\partial L/\partial\text{box}\)」的反向属 autograd 合集（todos #65）范围。

## 完整推导与下标约定

**下标约定**：行号 \(i\) 是第 \(i\) 对框，\(p=(x_1^p,y_1^p,x_2^p,y_2^p)\) 为预测框、\(t\) 为目标框；\(w_p=x_2^p-x_1^p\)、\(h_p=y_2^p-y_1^p\)，\(A_p=w_ph_p\)（\(t\) 同理）；闭包 \(C\) 的宽高记 \(w_C,h_C\)。全程逐对独立，不存在跨对的归约（除最终 mean）。

**union ≤ C面积 恒成立**：闭包在 x 方向覆盖两框的全部投影、在 y 方向同理，因此两框都被 \(C\) 完整包含，\(A_p+A_t\) 至多是「\(C\) 的面积加上被重复数了一次的交集」，即 \(A_p+A_t\le C\text{面积}+\text{inter}\)，移项正是 \(C\text{面积}-\text{union}\ge0\)。等号在一框把另一框"接齐"（两框并集恰好铺满闭包，例如两框同宽且上下相接）时成立——此时 GIoU 与 IoU 重合。于是 GIoU 的惩罚项非负、\(\text{GIoU}\le\text{IoU}\)，且不相交时（\(\text{inter}=0\)）
\[\text{GIoU}=-\frac{C\text{面积}-A_p-A_t}{C\text{面积}}\in[-1,0),\qquad \text{loss}_{giou}\in(1,2].\]
这就是示例 pair 0 的 1.0794 > 1：两框只交出一个 \(1\times1\) 的小角，闭包 \(3\times3\) 里还剩 2/9 的空洞没被覆盖。**GIoU 取到 \(-1\)** 的极限是两框都退化且互不相干（空洞占满闭包），这也是 \([ -1,1]\) 区间的下界证据。

**DIoU 的惩罚有界**：两中心都落在闭包内部，故 \(|\Delta c_x|\le w_C\)、\(|\Delta c_y|\le h_C\)，\(\rho^2\le w_C^2+h_C^2=c^2\)，惩罚项 ∈ \([0,1]\)，\(\text{DIoU}\in[-1,1]\)。中心式的具体写法是 \(\big(\tfrac{x_1^p+x_2^p}{2}-\tfrac{x_1^t+x_2^t}{2}\big)^2\)，题解用 \(((x_1^p+x_2^p-x_1^t-x_2^t)\cdot0.5)^2\) 是同一件事的少一次张量的写法。

**CIoU 的两个退化点**：
1. \(v=0\iff\) 两框宽高比相等（\(\arctan\) 单射，\(w_t/h_t=w_p/h_p\) 即可，正方形只是探针里的特例）。此时 \(\alpha v=0\)、CIoU 塌回 DIoU——判题用「两组全正方形」断言 loss_ciou 与 loss_diou 逐对相等。
2. \(\text{IoU}=1\iff\) 两框完全重合 \(\Rightarrow v=0\Rightarrow(1-\text{IoU})+v=0\)。这一分支**真的会被走到**（判题有 pred==target 用例），若直接写 `v/den` 就得 0/0 = NaN，`alpha` 必须显式兜成 0；否则三个损失在"完美预测"处反而变 NaN——这是本题最容易翻车的一处。

**除零兜底的分工**：union、C面积、\(c^2\) 三个分母在「合法框」承诺下恒正，兜底分支是给空框留的口；\((1-\text{IoU})+v\) 则如上所述必然可达。aspect 除法前的 `clamp_min(1e-7)` 同理是防"高为 0 的扁框"把 \(\arctan(w/0)\) 变成 \(\arctan(\infty)\)（虽然 \(\arctan(\text{Inf})=\pi/2\) 仍有限，但 Inf 会顺着差值污染 \(v\)，clamp 后 \(w/h\) 至多 \(10^7\)，数值完全可控）。

**统一取有限值的向量化写法**：题解把「mask 处才除、否则记 0」抽成一个 helper——先把非 mask 处的分母换成 1，再 `torch.where` 选结果。原因是 torch 的 `a/b` 会**先算出整个商张量**再选，分母里的 0 会当场产出 NaN/Inf；`torch.where` 只选值、不阻止分支被计算，所以必须预处理分母，而不是事后 where。

## 核心与易错点

1. **重合框的 \(\alpha\) 必须兜 0**：见上文退化点 2，这是唯一的"合法输入也会踩"的除零。
2. **三损失顺序固定** `(giou, diou, ciou)`，且都是 \(1-\)相似度而不是相似度；写成相似度会让 pred==target 的 0 探针爆掉。
3. **\(c^2\) 是对角线平方** \(w_C^2+h_C^2\)，不是 \(2(w_C+h_C)\)、也不是闭包面积；写错就破坏 \(\rho^2/c^2\le1\) 的有界性。
4. **GIoU 用的是闭包面积减 union，不是减 inter**；DIoU/CIoU 用的是同一个闭包的对角线平方，不要另算一个外接圆。
5. **中心距用 \(\rho^2\)（平方）而非 \(\rho\)**：论文式是平方比，开根会破坏与 \(c^2\) 的量纲一致，也和判题数值不同。
6. **N=0 + mean** 返回三个 0 标量，不是 NaN（`torch.mean` 在空张量上是 NaN，必须显式短路）。
7. **闭包宽高的 clamp_min(0)** 在合法框下取不到，但 \(\min/\max\) 一旦写反（例如 x2 用 min）就会造出负宽高，clamp 是最后一道防线。
8. **错误面**：`reduction` 非 mean/none（含 "sum"）必须 ValueError；返回必须是 3 元 tuple、mean 时 0 维、none 时 (N,)。

## 与邻近题的区别

- **vs `iou-compute`**：那题只算标量 IoU（含两种坐标口径），没有任何惩罚项；本题把闭包、中心距、宽高比三件一次算齐。基础 IoU 完全复用同口径，交集/并集部分可直接对照。
- **vs `nms`**：NMS 用 IoU 做**硬删除**（保留最高分、丢弃超阈值框），是推理期的选择算子；IoU 变体损失是训练期的**连续回归目标**，两者作用对象不同。
- **vs `soft-nms`（本题批次内）**：soft-nms 把 IoU 用作置信度的连续衰减因子（分数端），本题把 IoU 用作回归损失（坐标端）；两者共享 IoU 内核但一个改分数、一个改坐标。
- **vs `bce-with-logits` / `focal-loss` 等检测损失**：那些是分类损失（置信度/类别），本题是定位损失；真实检测头里两者相加。
- **vs autograd 合集（todos #65）**：那批做 \(\partial\text{loss}/\partial(x_1,y_1,x_2,y_2)\)，本题明确只到前向。

## 对拍口径与复杂度

- oracle `_oracle_pair` 用**纯 Python 标量**逐对重算：`max/min` 内置函数、`math.atan`、`math.pi`，与解法的张量 `torch.minimum/clamp_min/atan` 完全不同的实现路径；40 组固定种子随机框 × {none, mean} 全覆盖，浮点容差 \(10^{-5}\)。
- `_plain_iou_loss` 是**独立第二次实现**（不与 oracle 共享代码），只用于不等式探针 \(\text{loss}_{giou}\ge 1-\text{IoU}\)、\(\text{loss}_{diou}\ge 1-\text{IoU}\)、\(\text{loss}_{ciou}\ge\text{loss}_{diou}\)；这三条把三个变体的相对大小关系钉死，纯数值对拍漏掉的符号错误也能抓到。
- 判别性用例：pred==target（三损失恒 0，专杀 \(\alpha\) 的 NaN）、不相交且全正方形（loss_giou>1 且 loss_ciou==loss_diou）、随机框的变体大小关系、reduction 形状与 ValueError、N=0 的两个分支。
- 示例常量由参考实现实测回填：pair0/pair1 的六个逐对值与三个 mean 值（\(\text{mean}=(\text{pair0}+\text{pair1})/2\) 可在纸面复核）。
- 复杂度：时间 \(O(N)\)（每对常数次比较、乘法与两次 `atan`），额外空间 \(O(N)\) 量级的中间张量约十余个；`atan` 是唯一的超越函数，向量化后与逐对 python 循环相比只影响常数。
