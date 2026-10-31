# autograd 血缘账：叶、requires_grad 与反向图

## 题目描述

PyTorch 自动求导的全部「魔法」都写在一张元数据表上：一个张量要么是**叶**（`grad_fn is None`，用户造出来的、`optimizer.step()` 真正更新的那层），要么挂着一个**反向节点**（`grad_fn`，指向造出它的算子）；`requires_grad` 决定反向值不值得走；`.backward()` 沿 `next_functions` 把梯度送回叶的 `.grad` 缓冲；`no_grad` 与 `inference_mode` 都能「不建图」，但两者对已有的 `requires_grad`、对能否就地写、对 `detach_` 的效果并不一样。官方 *Autograd* 教程与 *Autograd mechanics* 笔记只给结论性的几条，真正逐语句的变化要动手试才知道。

本题把这块知识压成一套**语句小语言**：一段程序按顺序执行若干条语句，每条要么创建一个新张量、要么就地改动或读取已有张量；张量用语句下标寻址。要求逐条给出语句执行后的观测行——`requires_grad`、`is_leaf`、`is_inference`、秩、`grad_fn` 类名、视图的根、`.grad` 是否存在、以及这个缓冲是哪一次 `backward` 分配出来的。没有任何张量数值参与，全部是元数据推演。

## 规则

实现 `solve`（语句表、九列观测的定义与非法输入契约见 starter docstring）。下面只列口径要点：

1. **叶与节点互为反面**：`leaf == (node is None)`。数值/视图运算在求导态、且操作数里有一个「当前 `requires_grad` 为真且非 inference」的张量时才挂节点（add→`AddBackward0`、mul→`MulBackward0`、sum→`SumBackward0`、视图→`TBackward0`/`SelectBackward0`/`UnsqueezeBackward0`…）；`no_grad`/`inference_mode` 里结果一律不建图。inference 张量自己的 `requires_grad` 不驱动建图。
2. **视图与 `detach`**：视图的 `requires_grad` 与 `infer` 照抄操作数、**与 `ctx` 无关**（`no_grad` 里切出来的视图仍然要梯度，但因为没有 `grad_fn` 而成了叶），`view_of` 指向**根源**张量；`detach` 的 `requires_grad` 恒为假、且不再是视图。没拿到节点（当时不要梯度、或在 `no_grad`/`inference_mode` 里造出来）的视图**没有自己的元数据**，之后观测到的标志是**实时**读根源的。
3. **就地写**：`add_` 把主体拉进图并换成 `AddBackward0`（原本是叶也不再是叶）；「要梯度的叶」「要梯度的叶的视图」一律不许就地写。`detach_` 在视图上非法；inference 张量没有可复位的元数据，就地 `detach_` 关不掉它的标志。
4. **`requires_grad_`**：值与当前相同是静默空操作；不同值时只在叶上合法；inference 张量只关得掉、开不了（哪怕已经是真）。
5. **`backward`**：隐式梯度只对标量输出；主体当前必须真的要梯度，借标志的视图不行（引擎找不到节点也没有地方累加）。一次反向走过的图会被释放，但只有反向时要**回读保存张量**的节点（mul/sigmoid/relu）会让第二次反向失败；sum/add/neg/clone/视图这些不读回的节点可以被反复反向。梯度只落在叶引用与 `retain_grad` 过的中间张量上：`.grad` 为空就分配新缓冲（`grad_id` 记这次反向的语句下标），已有就累加进同一块；`grad_none` 清空后下一次反向会重新分配。
6. **读取类**：`grad` 把 `.grad` 缓冲本身当新张量上报（没有缓冲则失败）、`grad_none` 清空、`grad_zero_` 就地清零但不换缓冲、`probe` 只重新读一行。

## 示例

每行 `语句下标 语句 -> 观测`，`rg/leaf/infer/rank` 是四个元数据，`node` 是 `grad_fn` 类名，`base` 是视图的根源下标，`grad=#k` 表示这块缓冲是第 `k` 条语句（一次 `backward`）分配的，`fail` 表示这条语句被 torch 拒绝、程序状态不变。

链 1：反向消费图；第二次反向无论带不带 `retain_graph` 都不行，叶的缓冲只分配一次。

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, grad)    -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 2 sum(1, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 3 backward(2, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 backward(2, False) -> fail
 5 backward(2, True)  -> fail
 6 probe(0)           -> rg=1 leaf=1 infer=0 rank=2 grad=#3
```

链 2：同一条图带 `retain_graph=True` 可以反三次，梯度累加进**同一块**缓冲（`grad=#3` 始终不变）；`grad_none` 清空后再反才换手成 `grad=#9`，`grad_zero_` 只在原地清零。

```
 0 leaf(True, grad)  -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, grad)   -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 2 sum(1, grad)      -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 3 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#3
 5 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#3
 7 grad_none(0)      -> rg=1 leaf=1 infer=0 rank=2
 8 probe(0)          -> rg=1 leaf=1 infer=0 rank=2
 9 backward(2, True) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
10 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#9
11 grad_zero_(0)     -> rg=1 leaf=1 infer=0 rank=2 grad=#9
12 probe(0)          -> rg=1 leaf=1 infer=0 rank=2 grad=#9
```

链 3：`no_grad` 与 `inference_mode` 造出来的都是叶、都不要梯度；`inference_mode` 里却可以直接造一个 `requires_grad=True` 的叶，它照样不能反（没有图），`detach` 出来的新张量还带着 inference 身份。

```
 0 leaf(True, grad)           -> rg=1 leaf=1 infer=0 rank=2
 1 mul(0, 0, no_grad)         -> rg=0 leaf=1 infer=0 rank=2
 2 mul(0, 0, inference_mode)  -> rg=0 leaf=1 infer=1 rank=2
 3 leaf(True, inference_mode) -> rg=1 leaf=1 infer=1 rank=2
 4 detach(3, grad)            -> rg=0 leaf=1 infer=1 rank=2
 5 backward(3, False)         -> fail
 6 neg(3, inference_mode)     -> rg=0 leaf=1 infer=1 rank=2
 7 probe(4)                   -> rg=0 leaf=1 infer=1 rank=2
```

链 4：`no_grad` 里的视图不要节点、却借来根的标志；`add_` 不许写「要梯度的叶的视图」，`detach_` 不许写视图，而写普通叶是合法的——叶标志关了，已经拿到的缓冲还在。

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 t(0, grad)         -> rg=1 leaf=0 infer=0 rank=2 node=TBackward0 base=0
 2 index(1, grad)     -> rg=1 leaf=0 infer=0 rank=1 node=SelectBackward0 base=0
 3 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 4 backward(3, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 t(0, no_grad)      -> rg=1 leaf=1 infer=0 rank=2 base=0
 6 index(5, no_grad)  -> rg=1 leaf=1 infer=0 rank=1 base=0
 7 add_(5, 5)         -> fail
 8 add_(2, 2)         -> fail
 9 detach_(1)         -> fail
10 detach_(0)         -> rg=0 leaf=1 infer=0 rank=2 grad=#4
```

链 6：`retain_grad` 让中间张量也留下缓冲；非叶上的同值 `requires_grad_(True)` 是空操作；叶关标志不动缓冲。

```
 0 leaf(False, grad)        -> rg=0 leaf=1 infer=0 rank=2
 1 requires_grad_(0, True)  -> rg=1 leaf=1 infer=0 rank=2
 2 mul(0, 0, grad)          -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 3 retain_grad(2)           -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 4 sum(2, grad)             -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 backward(4, False)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(2)                 -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0 grad=#5
 7 grad(2)                  -> rg=0 leaf=1 infer=0 rank=2
 8 requires_grad_(2, True)  -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0 grad=#5
 9 probe(0)                 -> rg=1 leaf=1 infer=0 rank=2 grad=#5
10 requires_grad_(0, False) -> rg=0 leaf=1 infer=0 rank=2 grad=#5
11 probe(0)                 -> rg=0 leaf=1 infer=0 rank=2 grad=#5
```

链 10：`add_` 把旧节点留在新节点的入边里，所以第二个叶也能拿到梯度；写完再反一次，图已经是新的一张。

```
 0 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 1 leaf(True, grad)   -> rg=1 leaf=1 infer=0 rank=2
 2 mul(0, 0, grad)    -> rg=1 leaf=0 infer=0 rank=2 node=MulBackward0
 3 add_(2, 1)         -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
 4 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 5 backward(4, True)  -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 6 probe(1)           -> rg=1 leaf=1 infer=0 rank=2 grad=#5
 7 add_(2, 0)         -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
 8 sum(2, grad)       -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
 9 backward(8, False) -> rg=1 leaf=0 infer=0 rank=0 node=SumBackward0
10 probe(2)           -> rg=1 leaf=0 infer=0 rank=2 node=AddBackward0
```

推演链 4 的第 5、6 行：`t(0, no_grad)` 拿到的视图没有 `grad_fn`，`is_leaf` 为真；它的 `requires_grad` 是「借」来的——根张量此刻要梯度，所以读出来是 1。第 7 行 `add_(5, 5)` 因此撞上「叶的视图不能就地写」；第 10 行 `detach_(0)` 作用在普通叶上，关掉标志的同时保留第 4 行那次反向分配的缓冲。

## 约束

- 语句数 ≤ 40；张量下标从 0 开始，只引用成功创建张量的语句。
- 观测列全是布尔、整数与非空字符串，精确对拍；没有浮点比较，也没有随机数。
- 不得用 torch（或 numpy）真的执行语句再读元数据来查表得答案。
- 非法输入（不是列表、语句格式不对、下标越界、`flag` 不是 bool 等）抛 `ValueError`，与「语句被 torch 拒绝」是两回事；后者在观测行里报失败。
- 题目保证见 starter docstring：操作数形状一定可广播、不出现「inference 且 `requires_grad`」的张量当操作数、`requires_grad_` 的主体一定不是视图、反向失败时不会半途分配过梯度。
