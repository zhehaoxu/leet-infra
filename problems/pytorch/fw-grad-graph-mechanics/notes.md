# 详解：autograd 图机制账——叶子、detach、禁用模式与梯度累加

## 出处

PyTorch 文档 *Autograd Mechanism* 与 `torch.autograd` FAQ：

- **叶子判定**：`requires_grad=False` 的张量与用户显式 `requires_grad_(True)` 的张量是叶子；任何由可导运算产生的中间量都不是。图输入端的叶子对应的反向节点是 `AccumulateGrad`——这就是 `.grad` 被写入的位置。
- **`detach`**：返回与原张量**共享同一 storage** 的新张量，且它是 `requires_grad=False` 的叶子。`clone` 则新开 storage——把入参挂进自己的图时必须用 `clone`（`detach().requires_grad_(True)` 会让叶子与入参同存储，语义上等于把调用方的内存注册进了自己的计算）。
- **`no_grad` vs `inference_mode`**：`no_grad` 里的运算产物 `requires_grad=False` 但**仍是普通张量**，之后可以对它 `requires_grad_(True)` 重新作为叶子；`inference_mode` 里的运算产物是 **inference tensor**，离开上下文后 `requires_grad_(True)` 直接抛 `RuntimeError: Setting requires_grad=True on inference tensor outside InferenceMode`——这是两套禁用模式的本质差异（no_grad 只是"这次别记图"，inference_mode 是"这张纸永远不能进图"）。
- **梯度累加**：`backward` 对叶子 `.grad` 做 `+=` 而非覆盖。非标量输出必须显式传 `gradient=` 参数；`retain_graph=False`（默认）在反传后释放图缓冲，第二次 `backward` 抛 `RuntimeError: Trying to backward through the graph a second time`。
- `.grad` 本身是 `requires_grad=False` 的叶子张量。

## 核心与易错点

1. **入参保真**：`x.clone().requires_grad_(True)` 与 `x.detach().requires_grad_(True)` 的账本值完全相同，但后者与入参共享 storage——账本里的 `w_copies_input` 字段（`w.data_ptr() != x.data_ptr()`）就是这两条路的分界探针，测试用它把 detach 路拦下。
2. **链的顺序**：`(w*2+1).sum()` 的输出端链是 `SumBackward0 → AddBackward0 → MulBackward0 → AccumulateGrad` 四项——`+1` 在乘法之后，所以 `AddBackward0` 插在中间；第一次写锚点时容易按「乘、加、和」漏掉它。`next_functions` 是 `(node, input_nr)` 元组列表，叶子端是 `AccumulateGrad`。
3. **梯度账与输入数值无关**：`dy/dw = 2` 是常数，`backward(ones)` 两次得到 `[2,2] → [4,4]`（累加证据：第二次不是 `[2,2]`）；`d*3` 的第二张图给 `d.grad = [3,3]`。随机数值输入下梯度账不变，这是 `value_independent_grads` 的断言依据。
4. **两次 backward 的参数差异**：第一次必须 `retain_graph=True`，第二次必须不传——删掉第一次的 retain 立刻抛错（8/9 用例挂），这正是 `retain_graph=False` 默认值的可观察面。
5. **inference_mode 的两个方向**：inference tensor 事后不能开梯度（`t_rg_error=True`）；反过来，`no_grad` 产物可以。`t_is_inference()` 是比 `requires_grad` 更强的判据。

## 变异测试（7 有效全抓 + 1 等价）

`clone→detach`（被 `w_copies_input` 探针抓）、系数 2→3、去掉 `retain_graph`、去掉 `no_grad`、`inference_mode→no_grad`、吞掉 `t_rg_error` 的 try/except 改硬编码、梯度累加改覆盖、`d_shared_storage` 恒 False。**等价变异**：把 `t_rg_error` 捕获逻辑硬编码为 `True`——真实行为对所有合法输入恒为 True，硬编码版观测等价（但这正是 starter 声明"不得手填常量"的原因：它绕过了步骤 8 的真实执行）。

## 复杂度

2–5 元素张量、两次反传，微秒级；成本全在引擎语义。
