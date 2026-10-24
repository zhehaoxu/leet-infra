# 组装 MLP Block 详解

## 这题练什么

用 `nn.Module` 把「全连接 → ReLU → 全连接」这个最典型的 FFN block 组装起来，理解 PyTorch 模块系统的三个约定。

## 结构口径

隐藏维 $\text{hidden}=\text{in\_features}\times\text{hidden\_multiplier}$：

$$ \text{fc1}:\ \text{in}\to\text{hidden},\quad \text{ReLU},\quad \text{fc2}:\ \text{hidden}\to\text{out} $$

`forward` 即 `self.fc2(self.act(self.fc1(x)))`。Transformer 里的 `Linear(d, 4d)→act→Linear(4d, d)` 就是这个形状（multiplier=4）。

## nn.Module 三条铁律

1. **先 `super().__init__()`**：`__init__` 里第一件事必须调它，否则后续给 `self.fc1=...` 赋值时模块注册表还没建，参数不会被登记。
2. **子模块直接赋给 `self.xxx`**：`nn.Linear`/`nn.ReLU` 实例作为属性赋值即被自动注册，`.parameters()`、`.to(device)`、`.state_dict()` 才会带上它们。放进普通 Python list 则不会被追踪。
3. **只写 `forward`，调 `__call__`**：外部用 `module(x)`（触发 `__call__`→hooks→`forward`），不要直接 `module.forward(x)`。

## 参数与初始化

`nn.Linear` 默认 Kaiming-uniform 初始化、含 bias。判题只关心结构与前向输出形状/数值一致，不比对随机权重——所以**不要自己覆盖 `_reset_parameters`**，除非题面要求。

## 校验口径

`hidden_multiplier < 1` 时 `solve` 抛 `ValueError`（隐藏维至少等于输入维）；这是负路径测试点。

## 关联

linear-layer 是单个 `nn.Linear` 的数学；nn-module-block 把它组合成可训练子网络；optimizer-training-loop 演示这些参数如何被更新。
