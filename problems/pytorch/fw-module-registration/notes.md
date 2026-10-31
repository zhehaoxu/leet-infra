# 详解：nn.Module 注册账——参数去重、buffer 持久性与注册盲区

## 出处

PyTorch 文档 *Module* 与源码 `torch/nn/modules/module.py` 的 `__setattr__` / `named_parameters` / `state_dict` 实现，torch 2.14.1 实测：

- **遍历顺序**：`named_parameters` / `named_buffers` / `state_dict` 都按「本层 `_parameters` → 本层 `_buffers` → 子模块 `_modules`」的顺序产出——`__init__` 里先赋值的 `nn.Linear` 子模块，其参数排在**后赋值的直接 Parameter 属性之后**。很多人以为按赋值顺序，实测是按容器类别分桶再各自保序。
- **共享参数去重**：同一 `Parameter` 对象赋给两个属性，`named_parameters()` 默认 `remove_duplicate=True`，只产出**先注册的名字**；`remove_duplicate=False` 两个都出。而 **`state_dict()` 不去重**——两个名字都在，`load_state_dict` 会把同一参数写两次（值相同所以无害，但键账多一条）。
- **buffer 持久性**：`persistent=False` 的 buffer 出现在 `named_buffers()` 但不进 `state_dict()`；对已注册的 buffer 属性重新赋一个普通 Tensor，注册关系**保留**（`__setattr__` 检测到名字已在 `_buffers` 里就替换值），仍然进 `state_dict`。`.to(dtype)` 会原地改写 buffer。
- **注册盲区**：普通 Python `list` 装 `nn.Module` **不会被注册**（`modules()` 收不到、参数也不进 `named_parameters`）——这就是 `ModuleList`/`ModuleDict` 存在的理由；`register_parameter(name, None)` 合法（占位），但该名字在 `named_parameters` 与 `state_dict` 里都**隐身**。
- `__setattr__` 对 `Parameter` 自动注册进 `_parameters`，对普通 Tensor 若名字未注册则只当普通属性。

## 核心与易错点

1. **`module_count=4` 的账**：根模块 + `lin1` + `blocks` + `blocks.0`。`loose` 列表里的 Linear 不在（`loose_in_modules=False`），`module_linear_count=2`。
2. **`state_dict` 键序**：`["p_first", "p_second", "running", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]`——本层参数（含重复键）→ 持久 buffer → 子模块；`scratch`（非持久）与 `placeholder`（None）双双缺席。
3. **`nodup - dedup = 1`**：唯一共享对贡献恰好一条差值，随机尺寸下恒成立。
4. **两个实验字段**：`buffer_reassign_kept`（重新赋普通 Tensor 后注册仍在）与 `buffer_follows_to`（`.to(float64)` 后 buffer dtype 跟随）都为 True，对应 `__setattr__` 与 `_apply` 的真实实现。
5. **bool 是 int 子类**：`solve(True, 1, 1)` 必须拒绝——`isinstance(v, bool)` 的显式检查不可省。

## 变异测试（9/9 全抓）

去重账改成不去重、`dedup_keeps_first` 反转、`scratch` 改持久、`placeholder` 换真参数、`loose` 改 `ModuleList`、两个实验字段硬编码 False、去掉 `.to()`、去掉 bool 校验、`p_second` 换成独立对象。全部被锚点/盲区/随机用例抓获，无等价变异。

## 复杂度

O(参数量) 的注册账；宽度和 ≤ 24 时微秒级。
