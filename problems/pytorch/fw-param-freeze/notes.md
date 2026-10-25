# 详解：参数冻结、换 head 与优化器分组

## 出处

PyTorch 文档 `torch.optim.Optimizer`：优化器只认识构造时传入的参数张量；`torch.nn.Module.requires_grad_(False)`（`Tensor.requires_grad_`）只改变 autograd 行为，**不会**把参数从 `model.parameters()` 里移除。文档 FAQ「How do I checkpoint/restore only selected parts of a model」与迁移学习教程给出的正是「冻结骨干 + 换新 head + `[p for p in model.parameters() if p.requires_grad]`」这套配方。SGD 的缓冲区在 `torch/csrc/adam|sgd` 内核里按 `buf = momentum*buf + grad; param -= lr*buf`（dampening=0）就地更新。

## 核心与易错点

1. **冻结≠移除**：`total_before=124`（冻结前后总数不变），`trainable_before=0`。若把全部 `model.parameters()` 直接丢给 SGD，冻结参数虽然 `grad=None` 会被 step 跳过，但**已经传进 param_groups**，`load_state_dict` 恢复优化器时会多收 4 组空状态——正确习惯是过滤 `requires_grad`。
2. **换 head 改变参数量账**：\(6\times4+4=28 \to 6\times5+5=35\)，总数 \(124\to131\)。用 `named_parameters()` 顺序记 flags，`encoder.2`（Sequential 下标 2 的 Linear）不会被编号成 1/2/3。
3. **冻结参数 `.grad is None`**：requires_grad=False 的叶子不在反向图里，`zero_grad` 后也不会被填——这是 `frozen_grad_is_none=True` 的来源，也是「冻结层不占显存梯度」的机理。
4. **`len(opt.state)`**：优化器状态是首次 `step()` 时惰性创建的——`SGD(momentum=0.9)` 会为每个传入参数建一条动量缓冲区（2 个可训参数 → 2 条 state）；普通无 momentum 的 SGD 什么都不用存，实测 `len(opt.state)==0`。该计数同时反映两件事：只传了可训参数、且 momentum 确实启用了。
5. **逐位不变的判据**：必须用 `torch.equal` 而非 allclose——冻结参数没有理由差一个 ULP。

## 对拍口径

solution 走完整 autograd+优化器；判题 `_oracle` 不碰 `nn`/autograd：纯 Python 双循环重算前向（`math.tanh`）、解析梯度 \(\partial L/\partial W_{ij}=\frac{2}{N}\sum_b (f(x_b)-y_b)_i h_{bj}\)、手写动量递推，3 步后与 `final_head_*` 对拍（atol/rtol=1e-5）；计数、flags、布尔走硬编码精确比对。另设换头参数量差 7、首步动量过冲回升等断言。

## 复杂度

模型极小（131 参数、2 样本、3 步），全流程微秒级；oracle 双循环 \(O(B\cdot(48+36+30)\cdot 3)\) 瞬时。
