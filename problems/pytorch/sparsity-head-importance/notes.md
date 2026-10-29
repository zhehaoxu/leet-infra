# 详解：注意力头重要性打分与结构化剪枝

## 出处
头级结构化剪枝的代表工作把「一颗头」当作最小可删单元（Michel et al., *Are Sixteen Heads Really Better than One?*, NeurIPS 2019, arXiv 1905.10650），显著度可以用一阶 Taylor 的 \(|\,\partial\mathcal{L}/\partial g_h\cdot g_h\,|\)，也可以用本节这种免梯度的权重–激活统计量。权重–激活乘积 \(\|W_h\|_F\|X_h\|_F\) 是 **Wanda**（Sun et al., *A Simple and Effective Pruning Approach for Large Language Models*, 2023, arXiv 2306.11695）里 \(|W_{ij}|\cdot\|X_j\|_2\) 在 head 粒度上的推广：Wanda 按列（输入通道）打分后逐元素剪，这里按头切片打分后整片剪。

## 核心与易错点
1. **两个范数的口径**：\(\|W_h\|_F\) 只在该头自己的 \((d_\text{head},D)\) 片上求，\(\|X_h\|_F\) 只在 \((N,d_\text{head})\) 上求；混进别的头或把 \(N\) 也当自由维之外的批维都会算错。
2. **乘积而非加权和**：两者都是「越大越重要」，乘法天然让「一边为 0」直接判死——校准激活全零的头，无论权重多大都该先剪；若写成 \(a\|W\|+b\|X\|\) 就需要调参，语义也不对。
3. **全局排序**：剪枝比例是**全部头**的比例，不同层头数不同，不能按层各剪 `round(sparsity*H)`。层级结构在实现上必须先 `flatten(2)` 再比较。
4. **整片置零**：结构化剪枝的意义在于整颗头消失，若只把小权重置零就退化成了非结构化剪枝，算力并不会省。
5. **并列确定性**：`torch.argsort(..., stable=True)` 保证同分时按展平索引升序，判题才能逐位对拍。

## 对拍口径
参考解用 `flatten(2).norm(dim=2)` 分别求两个范数、相乘后 `argsort(stable=True)` 取前 \(k\) 个、再用 0/1 掩码乘回权重。判题 `_oracle` 用纯 Python `math.sqrt` 逐头求和、按 `(importance, 索引)` 排序后手工置零整片，属结构独立重算；逐位对拍 `importance` 与 `pruned_weight`。另设断言：被剪头数恰为 \(k\)（整片全零的头计数）、`sparsity=0` 完全不动、`sparsity=1` 全零、同分时剪展平索引更小的头、未剪头逐位不变、校准激活为 0 的头必被优先剪掉。

## 复杂度
\(O(L\cdot H\cdot(d_\text{head}D+N d_\text{head}))\) 求范数 + \(O(LH\log(LH))\) 排序；显存与输入同阶。
