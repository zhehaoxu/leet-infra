# 详解：为什么 gradcheck 必须穿过 autograd.Function

## 出处

`torch.autograd.gradcheck` 是框架岗的「你会验证自定义算子吗」的标准答案，但面试更常追问的是**它的实现细节**：为什么用中心差分、为什么步长取 \(\sqrt{eps}\)、为什么要逐元素而不是抽几个点。把这题写出来，等于把 gradcheck 的核心 60 行重写了一遍。

## 步长与误差的三角权衡

单侧差分的截断误差是 \(O(\delta)\)、中心差分是 \(O(\delta^2)\)；而舍入误差两侧都是 \(O(eps\,|L|/\delta)\)。取 \(\delta=\sqrt{eps}\approx1.5\times10^{-8}\)（float64）时，中心差分的总误差约 \(O(eps^{3/2})\sim10^{-12}\)，是实践中最优点。本题再用 \(\delta_i=\sqrt{eps}(1+|x_i|)\) 处理大数值元素（\(L\) 对 \(x\) 的敏感度随 \(|x|\) 增长）。相对误差的分母取 \(\max(|a|,|n|,10^{-8})\) 则避免梯度接近 0 时比值爆炸。

## 「必须走 Function」的用意

若实现里写 `torch.autograd.grad(forward(*inputs), inputs)`，得到的是 PyTorch 对 `forward` 计算图的自动微分结果，与 `backward` 传进来的手写公式毫无关系——校验会 100% 通过，哪怕手写梯度全错。正确做法是把两个 callable 包进一个 `autograd.Function` 的子类（`forward` 静态方法里算值并 `save_for_backward`，`backward` 静态方法里调用户函数并把 `grad_output` 传进去），这样 `torch.autograd.grad` 走的就是**用户提供的反向**。

## 核心与易错点

- **逐元素扰动**：把 `flat[i]` 改成 `x+δ` 与 `x−δ` 各算一次 `forward`；只扰动第一个元素或整个张量加同一个 δ，都会漏检。
- **每个输入都要检**：`errors` 的长度必须等于 `len(inputs)`，三输入只错第三个的用例专门抓「循环只跑一次」。
- **不要污染入参**：扰动必须在 `.clone()` 的副本上做，函数返回后原 `inputs` 的值与 `requires_grad` 都不能变。
- **0 维校验**：`forward` 返回非标量要抛 ValueError，否则 `grad_output` 的语义（本题按标量 1.0）不成立。
- **不可微点**：`relu` 在 0 处解析梯度与差分本就可能差一个台阶，测试里的折点用例放宽阈值。

## 与邻近题的区别

`fw-autograd-function` 只要求把 Function 写出来；本题要求的是**验证器**本身，视角从「实现算子」换成「实现算子的测试」。`ag-softmax-backward` 等题给出的是梯度公式，这里给出的是判断公式对不对的流程。

## 对拍口径与复杂度

判题不用单一 oracle，而是双向用例：正确 backward 的误差必须小于阈值（二次函数要求 < 1e-10）、错误 backward 的误差必须大于阈值，并检查返回列表长度与逐项大小。复杂度 \(O(2\sum_k n_k)\) 次前向（每元素两次），因此测试刻意用小形状。
