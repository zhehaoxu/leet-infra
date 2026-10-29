# PEFT：UniLoRA 共享参数向量的索引分配——算法详解

## 1. 公式与出处

UniLoRA 出自 arXiv:2506.00799，核心思想是把模型中所有 LoRA 适配器的可训练参数压缩到**一条共享向量** \(\theta\in\mathbb R^D\)。PEFT 风格的每层低秩更新仍保持：

\[\Delta W = \frac{\alpha}{r}\,B A,\qquad A\in\mathbb R^{r\times d_\text{in}},\ B\in\mathbb R^{d_\text{out}\times r}.\]

不同之处在于 \(A/B\) 不是层的独立参数，而是从 \(\theta\) 中按固定随机排列取出的切片。本题只考单层的索引分配与合并；多层场景下，每层使用 `proj_seed + layer_id` 作为独立种子，确保同一 \(\theta\) 在不同层取到不重叠的索引块。

## 2. 索引分配细节

- 本层共需 \(r(d_\text{in}+d_\text{out})\) 个标量。
- `torch.randperm(D, generator=g)` 产生 \([0,D)\) 的一个排列；取前 \(r d_\text{in}\) 个作为 \(A\) 的元素，接下来 \(r d_\text{out}\) 个作为 \(B\) 的元素。
- 用 `generator.manual_seed(proj_seed + layer_id)` 保证可复现；不依赖全局随机状态。
- `A.reshape(r, d_in)` 与 `B.reshape(d_out, r)` 均为行优先 reshape。

## 3. 退化与判题考点

- \(\alpha=0\)：缩放为零，\(W'=W_0\)。
- 选中切片全零：\(\Delta W=0\)，\(W'=W_0\)。
- `layer_id` 改变导致排列不同，是模拟「全模型共享向量」时层间索引不重叠的机制。
- 输出只依赖 \(W_0,\theta,r,\alpha,\text{proj_seed},\text{layer_id}\)，无随机性注入，CPU float32 完全确定。

## 4. 与现有题的分工

- `peft-lora-forward`/`peft-lora-merge`：每层独立拥有 \(A/B\)。
- 本题：\(A/B\) 是共享向量的切片，强调索引分配与参数量共享。
