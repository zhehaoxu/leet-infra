# nn.Module 注册账：参数去重、buffer 持久性与注册盲区

## 题目描述

`nn.Module` 的参数收集、buffer 管理与 `state_dict` 全靠注册机制，但它的行为有五处反直觉：遍历顺序按容器分桶、共享参数只去重一半、非持久 buffer 会隐身、普通 list 是注册盲区、None 参数是合法占位。本题要求把一个混用各种注册方式的模块的账本完整读出来。

## 规则

实现 `solve(in_dim, hidden, out_dim)`（三个正整数宽度），按 starter 注释的结构搭建模块并返回账本 dict：

| 字段 | 锚点值（4, 6, 3） | 考点 |
|---|---|---|
| `param_names_dedup` | `["p_first", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]` | 本层参数先于子模块；共享参数只留先注册名 |
| `param_names_nodup` | 上面插入 `"p_second"` 共 6 个 | `remove_duplicate=False` 的差值恰为 1 |
| `buffer_names` | `["running", "scratch"]` | named_buffers 不管 persistent |
| `state_dict_keys` | `["p_first", "p_second", "running", "lin1.weight", "lin1.bias", "blocks.0.weight", "blocks.0.bias"]` | **不去重**；非持久与 None 均缺席 |
| `sd_has_duplicate_key` | `True` | state_dict 对共享参数出两个键 |
| `placeholder_in_named` / `placeholder_in_sd` | `False` / `False` | None 参数隐身 |
| `loose_in_modules` | `False` | 普通 list 不注册 |
| `module_count` / `module_linear_count` | `4` / `2` | 根 + lin1 + blocks + blocks.0 |
| `buffer_reassign_kept` / `buffer_follows_to` | `True` / `True` | 重新赋普通 Tensor 注册仍在；buffer 跟随 `.to()` |

关键要求：

1. 任一宽度不是正整数（含 bool、浮点、字符串、0、负数）时抛 `ValueError`。
2. 账本字段必须由真实模块产生，不得手填常量。

## 示例

```
输入：in_dim=4, hidden=6, out_dim=3
去重参数 5 个、不去重 6 个；模块 4 个（list 里的 Linear 不算）
state_dict 7 键（p_second 也在，scratch 与 placeholder 不在）
```

## 约束

- 宽度在 1–8 之间即可；账本的顺序性（参数先于子模块、键序）在任意尺寸下不变
- 随机尺寸用例会复验 `nodup - dedup == 1` 与键序前三位
