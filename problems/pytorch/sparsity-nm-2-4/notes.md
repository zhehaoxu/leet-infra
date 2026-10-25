# 详解：2:4 结构化 N:M 剪枝

## 出处
NVIDIA Ampere Sparse Tensor Core 的 2:4 半结构化模式（developer.nvidia.com cusparselt 博客；PyTorch `to_sparse_semi_structured`）：每连续 4 元素至多 2 个非零，硬件按固定元数据布局把 GEMM 提速近 2×。

## 核心与易错点
1. **分组只沿最后一维、连续 4 个一组**：不是跨行、不是每行整体 top-2；`reshape(-1,4)` 展平到组即可，行边界不影响组划分（前提最后一维 %4==0）。
2. **组内独立 top-2**：每组恰好留 2 剪 2，整体稀疏率恒 50%，与元素大小分布无关。
3. **比较 \(|w|\) 保留原值**：mask 选位、元素本身不动符号大小；别把保留值改成 1。
4. 每 4 元素组要连续、对齐——这正是它比非结构化更易被硬件利用的原因。

## 对拍口径
参考解 `g.abs().topk(2,dim=1)` → scatter mask → `g*mask`。判题 `_oracle` 逐组用 Python `sorted` by \(|w|\) 取前 2、其余置零（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「每组非零数恰为 2」「稀疏率=0.5」「组间独立（改一组不影响另一组）」断言。

## 复杂度
逐组 top-2，\(O(N)\)。
