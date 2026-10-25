# 详解：前向/反向 Hook：抓激活与改梯度

## 出处

PyTorch 文档 `torch.nn.Module.register_forward_hook` / `register_full_backward_hook` / `register_backward_hook`：forward hook 签名 `(module, input, output)`，**返回值非 None 时替换模块输出**（单输出时直接返回张量即可，返回 `(t,)` 元组反而会把输出变成 tuple）。`register_full_backward_hook` 签名 `(module, grad_input, grad_output)`，返回的元组**替换 grad_input** 并继续向上传播；文档明确它只影响流向输入的梯度路径，且要求模块至少一个输入/输出 requires_grad 才触发。legacy `register_backward_hook` 在 forward 含多个 autograd 节点时会发 FutureWarning 且可能漏掉部分 grad_input——本题让它只计数不改写，正是为了安全地考「两种 hook 都挂得上、都会计数」这一账。

## 核心与易错点

1. **替换输出要计入前向与反向两本账**：tanh 的 hook 把输出乘 2 后，前向 \(out=W_2\cdot(2\tanh(a_1))\)；反向穿过这个乘 2 节点，所以 \(dL/dt=2\,dL/dh\)，不能只改前向。
2. **backward hook 只改 grad_input**：fc1 的 `2*g+1` 改写命中 `x.grad`，但 `fc1.weight.grad = outer(dL/da1, x)`、`fc1.bias.grad = dL/da1` 走的是另一条边，不受污染——判题用「参数梯度仍等于解析值」反向钉死这一点。
3. **`gin` 的长度**：hook 里 `grad_input` 对应「该模块 Function 收到的输入梯度」，对 `F.linear` 只含对数据输入的 1 项（参数梯度不从这里过）；返回元组长度必须与 `gin` 一致，改长度会抛 `RuntimeError: hook 'hook' has changed the size of value`。
4. **forward hook 里看到的 `inp`** 是模块 forward 收到的实际入参：fc2 的 `inp[0]` 是替换后的 \(2\tanh\)，这就是「替换生效」探针；`acts["tanh"]` 记录的是 hook 收到的**原始** out。
5. **触发次数账**：一次前反向下五个计数全为 1。多跑一步或少挂一路都会破。

## 对拍口径

solution 用真 hook + autograd；判题 `_oracle` 是纯 Python `math.tanh` 双循环手推：前向 \(\to\) loss \(\to\) 显式链式法则（含替换节点的 ×2 与 hook 的 2g+1），逐张量对拍 `atol=rtol=1e-5`；fired 计数精确比对；另有「fc2_input = 2*tanh」「bias.grad 未被 2g+1 污染」「x.grad 不是原始值」三条不变量断言。

## 复杂度

3→2→1 的迷你网络单步前反向，微秒级；难点全在 hook 语义的方向性。
