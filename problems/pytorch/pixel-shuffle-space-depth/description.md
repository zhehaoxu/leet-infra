# pixel_shuffle 与 space-to-depth/depth-to-space

## 题目描述

sub-pixel convolution（ESPCN，Shi et al., CVPR 2016）用「通道换分辨率」完成上采样：把 \((N, C\cdot r^2, H, W)\) 的特征按周期重排成 \((N, C, H\cdot r, W\cdot r)\)——即 pixel_shuffle；反方向把空间块塞回通道（space-to-depth / pixel_unshuffle）完成下采样。这一对**双射重排**是超分辨率头、focus 类骨干的基础算子：体积严格不变，只是每个元素换了地址。本题在一个入口实现四个 mode：shuffle / unshuffle / s2d / d2s，全部用 view/reshape/permute 或显式索引完成（**禁止调用 F.pixel_shuffle / F.pixel_unshuffle**），并以 `torch.equal` 级别钉死 s2d ≡ `F.pixel_unshuffle`、d2s ≡ `F.pixel_shuffle`。CPU、float32/float64、全程确定性。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。入参 x 为 4 维 \((N,C,H,W)\) float 张量（非 4 维抛 `ValueError`）；r 为缩放因子（非 int——含 float 与 bool——或 \(r<1\) 抛 `ValueError`）；mode 以下四选一，其余取值抛 `ValueError`。
- **shuffle**：\((N, C\cdot r^2, H, W)\to(N, C, H\cdot r, W\cdot r)\)。语义同 `F.pixel_shuffle`：\(out[n,\,c,\,h\cdot r+i,\,w\cdot r+j]=x[n,\,c\cdot r^2+i\cdot r+j,\,h,\,w]\)（\(0\le i,j<r\)）。通道数不能被 \(r^2\) 整除抛 `ValueError`（空间维只放大、无整除要求）。
- **unshuffle**：shuffle 的逆，\((N,C,H,W)\to(N, C\cdot r^2, H/r, W/r)\)。语义同 `F.pixel_unshuffle`：\(out[n,\,c\cdot r^2+i\cdot r+j,\,h,\,w]=x[n,\,c,\,h\cdot r+i,\,w\cdot r+j]\)。H、W 不能被 r 整除抛 `ValueError`。
- **s2d**：space-to-depth，形状契约与整除要求同 unshuffle；在 channels-first 布局下与 `F.pixel_unshuffle` 完全等价（输出通道序同为 \(c\cdot r^2+i\cdot r+j\)，测试以 `torch.equal` 钉死）。
- **d2s**：s2d 的逆，与 `F.pixel_shuffle` 完全等价，通道数整除约束同 shuffle。
- **实现约束（本题考点）**：不得调用 `F.pixel_shuffle` / `F.pixel_unshuffle`，必须用 view/reshape/permute 或显式索引重排完成该双射。
- 其他约定：r==1 时四种 mode 均为严格恒等（`torch.equal(y,x)`）；往返恒等 unshuffle(shuffle(x))==x、d2s(s2d(x))==x（`torch.equal`）；允许非连续输入（如 transpose 得到的视图），结果必须与同值连续输入完全一致；返回张量与 x 同 dtype、不修改入参。

## 示例

主示例（数值取自 test_cases 的 `EXAMPLE_*` 常量）：x \((1,4,2,2)\)，四个通道依次为 \(\begin{bmatrix}1&2\\3&4\end{bmatrix}\)、\(\begin{bmatrix}5&6\\7&8\end{bmatrix}\)、\(\begin{bmatrix}9&10\\11&12\end{bmatrix}\)、\(\begin{bmatrix}13&14\\15&16\end{bmatrix}\)，r=2，mode=shuffle，输出 \((1,1,4,4)\)：

\[\begin{bmatrix}1&5&2&6\\9&13&10&14\\3&7&4&8\\11&15&12&16\end{bmatrix}\]

读法即手写口径 \(out[2h+i][2w+j]=ch[i\cdot2+j][h][w]\)：第 0 行由通道 0/1 的第 0 行交错（\([1,2]\) 与 \([5,6]\) 织成 \([1,5,2,6]\)），第 1 行来自通道 2/3 的第 0 行，第 2 行回到通道 0/1 的第 1 行——通道按块内「先行后列」序铺开。同一输入 mode=s2d：输出 \((1,16,1,1)\)，16 个通道恰为 \(1,2,\dots,16\) 依 \(c\cdot4+i\cdot2+j\) 序（与 unshuffle 逐位相同）；unshuffle/d2s 走反方向可把 shuffle/s2d 的输出严格还原。

## 约束

- r 须为 int 且 \(\ge1\)（bool 不算合法 r）；整除校验分侧：shuffle/d2s 查通道 \(C\bmod r^2\)，unshuffle/s2d 查 H、W \(\bmod r\)；非法 mode、非 4 维输入一律 `ValueError`。
- 对拍：6 组 \((n,c,h,w,r)\) 配置上与 `F.pixel_shuffle` / `F.pixel_unshuffle` `torch.equal` 逐位一致（四种 mode 同一配置全查，含 \(r=1,2,3\)、batch>1、非方形空间）；独立四层循环 oracle 逐目标坐标算来源下标，与向量化 reshape/permute 分路径；双射断言（输出是输入值的置换：展平后排序相等）；往返恒等与 r=1 恒等（`torch.equal`）；transpose 非连续视图与连续输入结果 `torch.equal`。
- 纯重排无浮点误差，判题不用 allclose、全部 `torch.equal`；测试数据由黄金比例哈希生成（无 RNG）。
