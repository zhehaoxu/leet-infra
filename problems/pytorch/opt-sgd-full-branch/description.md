# SGD 全分支更新账

`torch.optim.SGD` 名义上只有一条更新式，实际有四个旋钮彼此耦合：`nesterov` 与
`dampening` 互斥、`dampening` 在**首个动量步被忽略**、`maximize` 的取负发生在
`weight_decay` **之前**、`momentum == 0` 时 `dampening` 被静默丢弃。官方文档的
伪码只写了主干，这些分支要读源码才有。

本题按 torch 2.14.1 的 `torch/optim/sgd.py` 逐行复刻：`SGD.__init__` 的四道参数
校验 + `_single_tensor_sgd` 的单步路径，一次跑多步，返回能逐步对拍的账本。

实现 `solve(params, grads, lr, momentum=0.0, dampening=0.0, weight_decay=0.0,
nesterov=False, maximize=False, buffers=None)`。

## 输入

- `params`：任意形状的张量（float32 / float64）。
- `grads`：长度 `T` 的序列，每项是与 `params` 同形状的张量，或 `None`
  （表示这一步该参数**没有梯度**）。
- `lr` / `momentum` / `dampening` / `weight_decay`：标量。
- `nesterov` / `maximize`：布尔。
- `buffers`：进入本次调用时**已有**的动量缓冲；`None` 表示没有历史状态。

## 参数校验

四道门，按下列**顺序**抛 `ValueError`，文案就是官方那几句（`{}` 里是入参值）：

```
Invalid learning rate: {}
Invalid momentum value: {}
Invalid weight_decay value: {}
Nesterov momentum requires a momentum and zero dampening
```

顺序是 `lr` → `momentum` → `weight_decay` → `nesterov` 条件；`nesterov` 的条件是
`momentum <= 0 or dampening != 0`。`dampening` 本身**没有任何取值范围校验**（可以
大于 1 或为负），`momentum == 0` 时也不会因为 `dampening != 0` 报错。

## 单步更新

对第 `t` 步（`t = 0, 1, …`）：

1. `grads[t] is None`：**整步跳过**——不建缓冲、不动参数，`dirs[t]` 记 `None`。
2. `maximize` 为真：`g ← -grads[t]`（**取负先于 weight_decay**）。
3. `weight_decay != 0`：`g ← g + weight_decay · params`，这里 `params` 是**本步更新前**的值。
4. `momentum != 0` 时分支：
   - 还没有历史缓冲（`buffers is None` 且此前没建过）：`buf ← g` 的拷贝，**这一步忽略
     `dampening`**；
   - 已经有缓冲（包括调用方显式传入的**全零**缓冲）：`buf ← momentum · buf + (1 - dampening) · g`；
   - `nesterov` 为真：`step ← g + momentum · buf`，用的是**刚更新后**的 `buf`；
     否则 `step ← buf`。
5. `momentum == 0`：`step ← g`，缓冲恒为全零，`dampening` 完全不参与。
6. `params ← params - lr · step`。`lr == 0` 时参数不动，但缓冲**照常推进**。

## 返回

一个 dict，四个键：

- `params`：更新后的参数。
- `buffers`：最终的动量缓冲；`momentum == 0` 或一次都没建起来时是全零。
- `path`：长度 `T + 1` 的参数快照，`path[0]` 是入参的拷贝。
- `dirs`：长度 `T`，每步实际用到的步进方向（跳过步为 `None`）。

入参一律不原地修改；返回的四个字段各自独立，就地改 `dirs` 不会改到 `buffers`。

## 约束

- `T` 可以是 0：此时 `path` 只有一段，`params` 原样，`buffers` 全零（或入参副本）。
- 与 `torch.optim.SGD` 的逐步结果在 float64 下按 `1e-12` 对拍。
- 不原地修改 `params` / `grads` / `buffers`。

## 示例

```
输入：params=[1.0, -2.0], grads=[[0.5, -1.0], [1.0, 1.0]], lr=0.1, momentum=0.9, dampening=0.3
输出：params=[0.835, -1.88], buffers=[1.15, -0.2]
```

第一串梯度建起缓冲 `[0.5, -1.0]`；第二串走 `0.9·buf + 0.7·g` 得 `[1.15, -0.2]`。

```
输入：params=[1.0, 1.0], grads=[[1.0, -2.0]], lr=0.1, momentum=0.9, dampening=0.5
输出：dirs=[[1.0, -2.0]], buffers=[1.0, -2.0]
```

`dampening=0.5` 在这一步**没有出现**——首个动量步的缓冲就是梯度本身；把它换成
`buffers=[[0.0, 0.0]]` 再跑，方向才会变成 `[0.5, -1.0]`。
