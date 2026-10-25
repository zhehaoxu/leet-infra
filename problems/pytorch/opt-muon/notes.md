# 详解：Muon Newton-Schulz 正交化动量

## 出处
Keller Jordan, *Muon: MomentUm Orthogonalized by Newton-schulz*（2024，博客 kellerjordan.github.io/posts/muon；实现 KellerJordan/Muon，用于 Modded-NanoGPT 训练纪录）。本题对齐其单设备 `SingleDeviceMuon` + `zeropower_via_newtonschulz5`，并把原 bfloat16 迭代改 float32 以保证 CPU 判题确定性。

## 核心思想
SGD-动量给出低秩、各向异性的更新矩阵 \(G\)。Muon 把它"正交化"为最接近的半正交矩阵 \(UV^\top\)（其中 \(G=USV^\top\)），相当于把每个奇异值都拉成 1——对所有方向等幅更新，经验上显著提升矩阵层训练速度。但 SVD 慢，故用 **Newton-Schulz 迭代**近似矩阵的零次幂 \(G(G^\top G)^{-1/2}\)。

## quintic Newton-Schulz
系数 \((a,b,c)=(3.4445,-4.7750,2.0315)\) 是把迭代 \(X\mapsto aX+(bX+cX^3)_{\text{型 quintic}}\) 在 0 附近斜率做到最大、只跑 5 步就"够用"（不收敛到严格 1，奇异值落在 \(\sim[0.5,1.5]\)，实践无损）。迭代写成：
\[ A=XX^\top,\quad B=bA+cA^2,\quad X\leftarrow aX+BA. \]
为让 \(XX^\top\) 是方阵且谱范数 \(\le1\)：
- 行 > 列时先转置（保证 \(X\) 宽 \(\ge\) 高，\(XX^\top\) 尺寸 = 较小维），最后转置回去；
- 用 **Frobenius 范数**归一化（\(\lVert X\rVert_F\ge\) 谱范数，故除以它可确保谱范数 \(\le1\)，迭代稳定），加 \(10^{-7}\) 防零矩阵。

## 谱缩放
正交化后 \(O\) 的每奇异值 \(\approx1\)；对 \(R\times C\)（\(R>C\)）真正的半正交矩阵其 Frobenius 范数是 \(\sqrt{C}\)。乘 \(\max(1,R/C)^{0.5}\) 是把更新幅度对齐"行多时更新也应更大"的 Adam 式尺度（KellerJordan 的 `adjust_lr` 约定）。

## Nesterov 组合的易错点
```
buf = μ·buf + (1−μ)·g
G   = (1−μ)·g + μ·buf     # 用的是更新后的 buf
```
即 nesterov 版：\(G=(1-\mu)g+\mu\,buf_{new}\)。若漏掉 \(g\) 的那半项、或用旧 buf，结果不同。

## 对拍口径
外部参照用等价向量化重算（同 quintic 迭代、同转置分支、同 Frobenius 归一化）。非方阵（\(5\times2\)、\(2\times3\)）都覆盖转置/不转置两分支。float32 下同算子序逐元素一致（示例值实测 maxdiff ≈ 1e-7，容差 1e-5 稳过）。`grad is None` 跳过。

## 注意
Muon 只应作用于隐藏层的二维权重矩阵；嵌入、输出头、标量/偏置建议仍用 AdamW——所以本题输入限定为二维矩阵。
