# 详解：Swin 窗口划分 Window Partition

## 出处
Liu et al., *Swin Transformer*（arXiv:2111.03861）`window_partition`：把 \((B,H,W,C)\) 特征图切成不重叠的 \((w,w)\) 窗口、展平成窗口批以做窗口内自注意力。是纯内存重排。

## 核心与易错点
1. **窗口顺序 (batch, 窗行, 窗列) 行主序**：`view(B, H//w, w, W//w, w, C).permute(0,1,3,2,4,5).reshape(-1,w,w,C)`——permute 把「窗行、窗列」挪到「窗内行、窗内列」之前。
2. **窗内保留 (row, col, C) 空间布局**：不是把窗口展平成 1D token，仍是 \(w\times w\times C\) 小特征图。
3. **必须整除**：Swin 原实现对不整除会 pad；本题约定 \(H,W\) 是 `window` 倍数、**不 padding**。
4. 与 patchify 的区别：这里最后维是通道 C、窗口保留 2-D 形状；patchify 把 (C·p·p) 全展平成特征。

## 对拍口径
参考解 `view→permute(0,1,3,2,4,5)→reshape`。判题 `_oracle` 用四重 Python 循环按窗口坐标逐元素填（结构不同、数值等价），`allclose`。另设「输出 shape = (B·(H/w)(W/w), w,w,C)」「window merge 逆还原 == 原图」「窗口 0 内容」断言。

## 复杂度
\(O(BHWC)\) 重排。
