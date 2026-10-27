# unfold/fold/roll：im2col、col2im 重叠累加与循环移位

## 题目描述

图像重排三兄弟收进一个入口：`unfold` 把每个滑窗感受野摊平成列——即 im2col，把卷积化成一次 GEMM 的经典预处理（Caffe 卷积的骨架，torch `F.unfold` 的官方语义）；`fold` 是它的反向 col2im——把列摊回图像画布，**重叠位置求和累加**（torch `F.fold`：combine local blocks "by summing the overlapping values"）；`roll` 做循环移位，越界元素绕回头部。三者的难点全在行为规格的细节账上：unfold 窗口的行/列排列（通道最外层的行主序）、fold 的重叠累加账、roll 的取模语义——以及本题与原生 `torch.roll` 的两处**刻意约定差异**（`dims=None` 不展平、int+元组只 roll 首维）。CPU、全程确定性，三种模式都返回新张量。

## 规则

- 完整签名、参数含义与默认值只写在 starter 的 docstring 里，那是唯一判题口径。`mode` 只取 `"unfold"` / `"fold"` / `"roll"`，其他取值抛 `ValueError`。
- 通用参数：kernel_size / stride / dilation / padding 均接受 int 或 \((h,w)\) 二元组；**stride=None 时取 kernel_size**（本题约定，unfold/fold 都适用；注意原生 F.unfold/F.fold 的默认 stride 是 1）。窗口数 \(H_{out}=(H+2p_h-d_h(k_h-1)-1)\,//\,s_h+1\)（\(W_{out}\) 同理），\(L=H_{out}\cdot W_{out}\)。
- **unfold（im2col）**：x 须为 \((N,C,H,W)\) 四维（非 4 维抛 `ValueError`），先零填充再滑窗，返回 cols \((N,\,C\cdot k_h k_w,\,L)\)。**行按 \(c\to k_h\to k_w\) 行主序**：\(row=c\cdot(k_h k_w)+i\cdot k_w+j\)（通道最外层、窗口内先高后宽）；**列按 \(hi\to wi\)**：\(l=hi\cdot W_{out}+wi\)——与 `F.unfold` 实测一致。落在输入边界外的窗口元素由零填充提供。\(H_{out}\) 或 \(W_{out}\le 0\)（kernel 大于 pad 后尺寸）抛 `RuntimeError`（与原生一致）。
- **fold（col2im）**：x 此时是 cols \((N,\,C\cdot k_h k_w,\,L)\) 三维（非 3 维抛 `ValueError`），output_size=\((H,W)\) 必填（缺省抛 `ValueError`），返回 \((N,C,H,W)\)。**重叠位置求和累加**；\(C=\text{shape}[1]/(k_h k_w)\)（须恰能整除）。shape[1] 不被 \(k_h k_w\) 整除、或 L 不恰等于按 output_size 算出的 \(H_{out}W_{out}\)（output_size 过大或过小都算不匹配）抛 `RuntimeError`（与原生一致）。
- **roll**：循环移位，逐维 \((i+shift)\bmod n\) 语义（取模，支持负 shifts）；kernel_size / stride / dilation / padding / output_size 全部忽略。shifts 必填（None 抛 `ValueError`）：
  - shifts 为 int、dims 为 None：对**最后一维** roll（本题约定；注意原生 `torch.roll` 在 dims=None 时是先展平再 roll，本题**不展平**）；
  - shifts 为 int、dims 为 int：对该维 roll；
  - shifts 为 int、dims 为元组：**只对 dims[0] 这一维** roll（本题约定；原生 `torch.roll` 不接受这种组合）；
  - shifts 为元组：dims 必须是等长元组，逐维分别 roll；dims 为 None 或长度不匹配抛 `ValueError`；
  - dims 越界抛 `IndexError`（负维先按 \(+ndim\) 归一化）。
- 三种模式都返回**新张量**，不修改输入；除上述注明 `ValueError` / `IndexError` / `RuntimeError` 的情形外，不做额外校验。

## 示例

主示例（数值取自 test_cases 已回填的 `EXAMPLE_*` 常量）：单通道 \(2\times3\) 图 \(x=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\)，kernel=2、stride=1、mode=unfold：

\[cols=\begin{bmatrix}1&2\\2&3\\4&5\\5&6\end{bmatrix}\]

每个窗口一列，列内 4 行按核元素 \((i,j)=(0,0),(0,1),(1,0),(1,1)\) 排（左上、右上、左下、右下）。同一输入上 roll：\([1,2,3,4,5]\) shifts=2（dims 缺省 → 最后一维）\(\to[4,5,1,2,3]\)。

窗口排列与零填充（2 通道手写案例，把行序钉死）：\(x=\mathrm{arange}(24)\) reshape \((1,2,3,4)\)、k=2、s=1 → cols \((1,8,6)\)，窗口 \((0,0)\) 一列为 \([0,1,4,5\,|\,12,13,16,17]\)——c0 的 \(\begin{bmatrix}0&1\\4&5\end{bmatrix}\) 在前 4 行、c1 的 \(\begin{bmatrix}12&13\\16&17\end{bmatrix}\) 在后 4 行。零填充窗口：\(x=1..12\) reshape \((1,1,3,4)\)、k=3、s=1、pad=1，窗口 \((0,0)\) 一列为 \([0,0,0,0,1,2,0,5,6]\)（核 \(3\times3\) 行主序摊平，越界补 0）。膨胀跳格：同图 k=2、d=2 → cols \((1,4,2)=\begin{bmatrix}1&2\\3&4\\9&10\\11&12\end{bmatrix}\)。

fold 重叠累加账：全 1 cols \((1,4,6)\)、k=2、s=1、output_size=\((3,4)\) →

\[\begin{bmatrix}1&2&2&1\\2&4&4&2\\1&2&2&1\end{bmatrix}\]

角被 1 个窗口覆盖、边被 2 个、中心被 4 个——重叠区是**求和**不是覆盖。

roll 约定与原生对照：\(\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\) shifts=1、dims 缺省 → \(\begin{bmatrix}3&1&2\\6&4&5\end{bmatrix}\)（逐行最后一维 roll）；原生 `torch.roll` 是**先展平再 roll**，得 \(\begin{bmatrix}6&1&2\\3&4&5\end{bmatrix}\)——刻意不同，测试把两个结果都 assert 钉死。shifts=1、dims=\((0,1)\) → 只 roll 第 0 维 → \(\begin{bmatrix}4&5&6\\1&2&3\end{bmatrix}\)。

往返恒等：stride=kernel、padding=0（且空间维可整除）时 fold∘unfold 严格恒等（`torch.equal` 复原输入）；padding>0 时只要 pad 后尺寸仍被 kernel 整除，裁边后同样恒等。

## 约束

- 数值对拍：unfold/fold 与原生 `F.unfold`/`F.fold` 在 7 组 \((k,s,d,p)\) 组合（含二元组、非对称 padding/dilation）× 2 种输入形状上 `allclose(1e-6)`；stride=None → kernel_size 的约定单独对拍（原生需显式传 stride 对齐）。roll 与 `torch.roll` 在 13 组 shifts/dims（含负维、超长 shift）上 `torch.equal`。
- 另有独立多重循环 oracle：unfold 逐窗口逐核元素写列（含零填充边界判断）、fold 逐窗口 `+=` 累加、roll 逐元素 \((pos-s)\bmod n\)，与向量化实现分路径对拍。
- 判别口径评测直接断言：窗口行/列排列（2 通道手写案例）、重叠求和（ones 账）、stride=kernel 往返恒等、roll 两处约定差异（与原生展平行为对照）、错误分类三件套（`ValueError` / `RuntimeError` / `IndexError`）。
- 返回张量不共享输入内存（新张量）；除 docstring 注明的校验外不做额外检查。
