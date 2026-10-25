# 详解：ViT 图像分块 Patchify

## 出处
Dosovitskiy et al., *An Image is Worth 16×16 Words*（arXiv:2010.11929）：把图像切成不重叠 patch 展平成 token 序列。本题只做几何重排（线性投影是后续 `nn.Linear`，不涉及）。

## 核心与易错点
1. **token 顺序 = 块行优先**：先遍历块行、再块列（`permute(0,2,4,...)` 把 block_row、block_col 提到前面），不是通道优先、也不是列优先。
2. **token 内展平顺序 (C, 行, 列)**：通道在最外维。不同约定会让后续投影权重错位——本题固定为 (c,i,j)。
3. **reshape 拆维要对应 H→(nr,p)、W→(nc,p)**：`reshape(B,C,nr,p,nc,p)` 依赖内存连续；对非连续输入先 `.contiguous()`。
4. 是**双射**：与 unpatchify 互为逆，元素既不丢也不重。

## 对拍口径
参考解 `reshape→permute(0,2,4,1,3,5)→reshape`。判题 `_oracle` 用五重 Python 循环逐元素按约定填入（结构完全不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「shape 正确」「unpatchify 逆还原 == 原图」「块数 N=(H/p)(W/p)」「单块数值」断言。

## 复杂度
\(O(BCHW)\) 重排。
