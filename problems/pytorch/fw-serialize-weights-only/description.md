# weights_only 白名单与 state_dict 键级账

## 题意

`torch.load` 自 2.6 起 `weights_only` **默认 True**：pickle 流里任何 GLOBAL 指令（自定义类/函数）不在白名单即 `UnpicklingError`。本题手写判定层——`torch.save` 的 zip 包里抽出 `archive/data.pkl`，`pickletools` 扫 GLOBAL 指令（protocol 2 全为两行文本 GLOBAL、无 STACK_GLOBAL），逐条对照题面钉死的允许键表判 ok/rejected，并手写 `load_state_dict` 的 missing/unexpected 键级账。不调 `torch.load` / `load_state_dict`。

## 求解

官方口径（`torch/serialization.py` + `torch/_weights_only_unpickler.py` v2.14.1）：

- 纯 primitive 容器（dict/list/tuple/int/float/str/None/bool）不产生 GLOBAL → 天然放行；
- 白名单（protocol-2 键形态，实测反推）：`collections OrderedDict/Counter`、`torch Size/device/<dtype>`、`torch <dtype>Storage`、`torch._utils _rebuild_tensor_v2/_rebuild_parameter`、`__builtin__ set/bytearray/complex`、`_codecs encode`；
- 典型拒绝：`torch.nn.modules.linear Linear`（任何 nn.Module）、`types SimpleNamespace`、**`__builtin__ xrange`（range 对象）**、numpy 系（`numpy dtype/ndarray`、`numpy._core.multiarray _reconstruct`）；
- lambda 在 **save 阶段**即 `PicklingError`（save_fail，非 load 拒）；
- `load_state_dict`：missing = 模型有而 state_dict 无、unexpected 反之，两者 **sorted 后**报出（字典序）；`strict=True` 且任一非空 → `RuntimeError`（首行 `Error(s) in loading state_dict for ...`）；`strict=False` → `IncompatibleKeys`；全匹配 → `<All keys matched successfully>`；
- `nn.Linear` 根模块的 state_dict 键是 `weight/bias`——无属性名前缀（`fc.` 只在容器模块层级出现）。

## 数据规模

- 17 个场景对象（tensor dict、嵌套 primitive、OrderedDict/Counter/Size/device/dtype/set/bytearray/complex/Parameter 放行组；Linear/SimpleNamespace/range/numpy 拒绝组；lambda save_fail）。
- state_dict 三账：全匹配、异构模型（`0.*` vs `weight/bias`）、带 `extra.weight` 冗余键。
