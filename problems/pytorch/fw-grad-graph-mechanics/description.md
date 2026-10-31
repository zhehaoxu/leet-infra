# autograd 图机制账：叶子、detach、禁用模式与梯度累加

## 题目描述

autograd 的动态图由「叶子 → 运算 → grad_fn 节点」组成。写训练代码绕不开四个机制问题：哪些张量是叶子？`detach` 到底共享了什么？`no_grad` 和 `inference_mode` 的产物有何本质不同？两次 `backward` 之间梯度是覆盖还是累加？本题把这四问变成一张必须逐项对账的账本。

## 规则

实现 `solve(x)`（`x` 为一维浮点张量，`requires_grad=False`），按指定 12 步建图并返回账本 dict（字段含义见 starter 注释）：

| 字段 | 锚点值（`x=[1.0, 2.0]`） | 考点 |
|---|---|---|
| `w_is_leaf` / `w_copies_input` | `True` / `True` | 叶子必须 `clone` 造，不得共享入参存储 |
| `y_is_leaf` / `y_grad_fn` | `False` / `"AddBackward0"` | 运算产物非叶，grad_fn 是最后一个算子 |
| `chain` | `["SumBackward0", "AddBackward0", "MulBackward0", "AccumulateGrad"]` | 输出端到叶子的节点链，`+1` 插在乘法与求和之间 |
| `leaf_accumulator` | `"AccumulateGrad"` | 叶子在图输入端的节点类名 |
| `d_is_leaf` / `d_requires_grad` / `d_shared_storage` | `True` / `False` / `True` | detach = 同 storage 新叶子 |
| `n_requires_grad` | `False` | `no_grad` 产物不可导 |
| `t_requires_grad` / `t_is_inference` / `t_rg_error` | `False` / `True` / `True` | inference tensor 事后不能开梯度 |
| `w_grad_first` / `w_grad_second` | `[2,2]` / `[4,4]` | 梯度累加而非覆盖 |
| `d_grad` | `[3,3]` | detach 产物可重新作为叶子建第二张图 |

关键要求：

1. 不得修改入参：值不变、`requires_grad` 保持 `False`。
2. `x` 不是一维浮点张量时抛 `ValueError`。
3. 步骤 9 的 `backward` 必须 `retain_graph=True`，步骤 10 不得传。

## 示例

```
输入：x = [1.0, 2.0]
y = w*2+1 = [3, 5]
w.grad：第一次 backward 后 [2,2]，第二次后 [4,4]   # 2*ones 累加两次
d.grad = [3, 3]                                   # (d*3).sum() 的第二张图
```

## 约束

- `x` 形状任意（一维），float32/float64 皆可；梯度账与数值无关（`dy/dw=2` 是常数）
- 判题对入参随机化：`requires_grad`、数值、`_version` 均不得改变
