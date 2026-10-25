# 详解：逐行（局部）幅值剪枝

## 出处
局部 / layer-wise 剪枝（相对于 Han et al. 全局阈值）：把稀疏预算在每个输出神经元（矩阵每一行）内独立分配，等价于 `torch.nn.utils.prune` 里按维度分组的 local 思路，工程上常用以保持各神经元激活均衡。

## 核心与易错点
1. **每行独立竞争本行预算**：与全局题 sparsity-magnitude-prune 唯一区别就是阈值作用域——这里第 \(i\) 行只和自己比，每行剪 \(k=\mathrm{round}(\text{sparsity}\cdot \text{in})\) 个。
2. **每行剪除数相同 → 每行稀疏率都等于 sparsity**：不会出现某行被剪空、某行不动。
3. **沿 dim=1 排序 + scatter 置零**：`argsort(dim=1,stable=True)[:k]` 再 `scatter_`；别展平成全局。
4. 只看 \(|w|\)、保留原值原符号；并列按列索引升序保确定性。

## 对拍口径
参考解 `abs().argsort(dim=1,stable=True)[:,:k]` → `scatter_(1,idx,0)`。判题 `_oracle` 逐行 Python `sorted` by \(|w|\) 保留后段、其余置零（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。另设「每行非零数一致=⌈in·(1-sparsity)⌉」「各行独立」「与全局题在跨行分布不同处结果有别」断言。

## 复杂度
\(O(N\log C)\)（逐行排序）。
