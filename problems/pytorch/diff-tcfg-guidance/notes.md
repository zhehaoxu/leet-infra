# 详解：TCFG 的 SVD 截断投影引导

## 出处

*Tangentially Corrected Classifier-Free Guidance*（arXiv:2503.18137）。实现口径对齐 HuggingFace diffusers 的 `src/diffusers/guiders/tangential_classifier_free_guidance.py`：`normalized_guidance`（L132–151）把两支预测堆成 `(B, 2, D)`、做 `torch.linalg.svd(..., full_matrices=False)`，把 `Vh` 的第二行置零后与 `uncond` 相乘，等价于只保留第一右奇异方向上的投影；外推基准是投影后的 `uncond`。

## 核心与易错点

1. **不是「把更新投到切平面」**：文档里常被概括成切向引导，但实现是把 \(\epsilon_u\) **本身**投到主方向上，再以投影点为基准外推。两种写法结果不同，必须以源码为准。
2. **只有两行，不必真做 SVD**：\(v_1\) 一定落在 \(\{\epsilon_c,\epsilon_u\}\) 内。令 \(v_1=\alpha\epsilon_c+\beta\epsilon_u\)，则 \((\alpha,\beta)\) 正是 \(2\times2\) Gram 矩阵 \([[\|c\|^2,\,c\!\cdot\!u],[c\!\cdot\!u,\,\|u\|^2]]\) 最大特征值对应的特征向量——因为该 Gram 矩阵就是 \(A^\top A\) 在基 \(\{\epsilon_c,\epsilon_u\}\)（非正交基）下的表示。闭式解与 `torch.linalg.svd` 在 200 组随机输入上逐位一致。
3. **奇异值并列时主方向不唯一**：\(p=q\) 且 \(c\cdot u=0\) 时两个奇异值相等，SVD 的取值依赖底层实现。本题约定取 \(\epsilon_c\) 方向，此时投影为 0。闭式解里表现为两个候选特征向量的系数同时为零，需要显式退化处理，否则会变成 0/0。
4. **投影用标量倍数即可**：\(p=\frac{\epsilon_u\cdot w}{w\cdot w}w\)，\(w\) 不必归一化；先归一化再点乘也能得到同样结果，但多一次除法、也更容易在 \(w=0\) 时炸。
5. **外推基准是投影点**：\(s=1\) 时 \(p+(\epsilon_c-p)=\epsilon_c\) 逐位成立，是最可靠的自检点。
6. **两支平行时退化成裸 CFG**：\(v_1\) 就是公共方向，投影不改变 \(\epsilon_u\)，所以结果与普通 CFG 完全一致——这也说明 TCFG 只在两支「方向不一致」时才起作用。

## 对拍口径

参考解走 \(2\times2\) Gram 的闭式特征分解（float64），完全没有调用 SVD；判题 `_oracle` 则照 diffusers 原样用 `torch.linalg.svd`（float64）算投影，两条实现路径互不依赖，随机对拍 `atol=1e-5`（实测差值为 0）。另外四组结构断言：\(s=1\) 恒等、\(s=0\) 时投影残差与投影正交、并列/平行/零向量三种退化、以及 \(\text{out}-p\) 关于 \(s\) 的线性。

## 复杂度

闭式路径逐样本 \(O(D)\)（只需内积与两次标量缩放）；若直接走 SVD 则是 \(O(D)\) 的 2 行分解，实现更短但常数更高。
