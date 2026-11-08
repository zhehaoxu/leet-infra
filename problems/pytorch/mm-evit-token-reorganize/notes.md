# EViT 源码解剖：token 重组织账

> 官方仓库：`youweiliang/evit`（默认分支 **`master`**，不是 `main`）
> 涉及文件：`evit.py`、`helpers.py`

## 1. 这套规则在 EViT 里的位置

EViT 是把「token 剪枝」塞进 ViT 的一个变体：每个 Block 里，注意力层会顺带
决定「这一层要保留哪些 patch token」，Block 后半段再把序列重新组织一遍。

三个官方函数各管一段：

| 官方位置 | 职责 |
|---|---|
| `evit.py::Attention.forward` | 算出 `cls_attn` 打分、用 `topk` 选出要保留的 token |
| `evit.py::Block.forward` | 按选出的 `idx` gather 出保留的 patch、把丢弃的压成 `extra_token`、拼接 |
| `helpers.py::complement_idx` | 求 `idx` 在 patch 空间里的补集 |

`Attention.forward` 的完整骨架（官方原文）：

```python
left_tokens = N - 1
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
    left_tokens = math.ceil(keep_rate * (N - 1))
    if tokens is not None:
        left_tokens = tokens
    if left_tokens == N - 1:
        return x, None, None, None, left_tokens
    assert left_tokens >= 1
    cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
    cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
    # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
    # index = torch.cat([cls_idx, idx + 1], dim=1)
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

    return x, index, idx, cls_attn, left_tokens

return  x, None, None, None, left_tokens
```

`Block.forward` 的重组段（官方原文）：

```python
tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
x = x + self.drop_path(tmp)

if index is not None:
    non_cls = x[:, 1:]
    x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

    if self.fuse_token:
        compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
        non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
        extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
    else:
        x = torch.cat([x[:, 0:1], x_others], dim=1)

x = x + self.drop_path(self.mlp(self.norm2(x)))
n_tokens = x.shape[1] - 1
```

---

## 2. `cls_attn`：打分口径

### ★ 要点 2.1：只取 attention 的**第 0 行**

`attn` 的形状是 `[B, H, N, N]`（query × key）。`attn[:, :, 0, 1:]` 里的
`0` 落在**query 维**上 —— 也就是「CLS 作为 query 去问所有 key」的那一行。
其余 N-1 行（patch 作为 query）**完全不参与**打分。

### ★ 要点 2.2：列从 **1** 开始 —— 去掉 CLS 自己那一列

`attn[b, h, 0, 0]` 是 CLS 看自己的注意力权重，恒为正且没有判别力，
所以被切掉了。因此 `cls_attn` 的长度是 **N-1** 而不是 N。

### ★ 要点 2.3：跨 head 求**算术平均**

```python
cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
```

`dim=1` 是 **head** 维（`attn[:, :, 0, 1:]` 之后的形状是 `[B, H, N-1]`）。
用 `mean` 而不是 `sum` 是因为不同 head 的注意力尺度不完全可比，
取平均相当于让各 head 各投一票。

实测（3 个 head × 5 个 token）：

```
cls_attn[0] = (0.1 + 0.3 + 0.1) / 3 = 0.16666...
```

---

## 3. `left_tokens`：数量决策的三重口径

### ★ 要点 3.1：触发条件里的两个变量**不是同一个东西**

官方那一行：

```python
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
```

- `self.keep_rate` 是 **构造时**写进 `Attention` 的 `self.keep_rate`；
- `keep_rate` 是 **调用时**传进来的参数（`forward` 的形参，缺省时
  `keep_rate = self.keep_rate`）。

Python 里 `and` 优先级高于 `or`，所以整行的语义是：

```python
(self.keep_rate < 1 and keep_rate < 1) or (tokens is not None)
```

**两个比率必须都 `< 1`** 才走 keep_rate 分支。但 `tokens` 可以**单独**触发。

实测：

| `self.keep_rate` | 调用时 `keep_rate` | `tokens` | 是否触发 |
|---|---|---|---|
| 0.5 | 0.5 | None | ✔ 触发 |
| **1.0** | 0.5 | None | ✘ **不触发**（`self.keep_rate < 1` 为 False） |
| 0.5 | 1.0 | None | ✘ 不触发（`keep_rate < 1` 为 False） |
| **1.0** | 0.5 | **3** | ✔ **触发**（`tokens is not None` 单独成立） |

这是官方注释 `# double check the keep rate` 想提醒的东西 ——
构造时写死 `keep_rate=1.0` 的模块，即使调用时传了 `keep_rate=0.5`，
**也不会**进入剪枝分支。

### ★ 要点 3.2：`left_tokens = math.ceil(keep_rate * (N - 1))`

- **向上取整**（`ceil`），不是 floor、不是 round；
- 基数是 **`N - 1`**（N 含 CLS token，即 patch 数）。

实测：`N = 197`（196 个 patch）、`keep_rate = 0.7` → `ceil(0.7 * 196) = 138`。

### ★ 要点 3.3：`tokens` 直接**覆盖**

```python
if tokens is not None:
    left_tokens = tokens
```

是**直接赋值**，不是 `min`、不是 `max`。`tokens` 比 ceil 结果大时照用。

### ★ 要点 3.4：`left_tokens == N - 1` 时**提前返回**

```python
if left_tokens == N - 1:
    return x, None, None, None, left_tokens
```

此时 **条件是触发了的**（`triggered` 仍为 `True`），但因为「要保留的正好
是全部 patch」，没必要再走 topk，`index` / `idx` 都是 `None`。

实测：`N = 10`、`keep_rate = 0.9` → `ceil(0.9 * 9) = 9 == N - 1` → 提前返回；
`keep_rate = 0.888` → `ceil(7.99) = 8` → 正常触发。

### ★ 要点 3.5：`assert left_tokens >= 1`

保留数量不足 1 个是非法的，官方直接断言失败。

---

## 4. `topk` 与 `complement_idx`

### ★ 要点 4.1：`topk` 的输出是**分数降序**

```python
_, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
```

`sorted=True` 意味着返回值**按分数从大到小**排列，**不是**按原下标升序。

实测：`cls_attn = [0.1, 0.5, 0.3, 0.05, 0.05]`、`left_tokens = 3`
→ `idx = [1, 2, 0]`（分数 0.5、0.3、0.1），而下标升序应是 `[0, 1, 2]`。

这个顺序会一路传到 `Block.forward` 的 gather —— 也就是说**保留下来的 patch
在序列里是按分数从高到低排的**，位置信息被重排了。

### ★ 要点 4.2：并列时按下标升序

`torch.topk` 在分数并列时按**下标升序**取。实测：

- `[0.5, 0.5, 0.5, 0.1]` topk(2) → `[0, 1]`
- `[0.1, 0.5, 0.5, 0.5]` topk(2) → `[1, 2]`
- `[0.3, 0.3, 0.3, 0.3]` topk(2) → `[0, 1]`

### ★ 要点 4.3：`complement_idx` 返回**按下标升序**的补集

官方实现（`helpers.py`）：

```python
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
```

关键在 `torch.sort(..., descending=False)` —— 结果是**升序**。

实测：`complement_idx([[1, 3]], dim=4)` → `[[0, 2]]`；
`complement_idx([[0, 1, 2]], dim=4)` → `[[3]]`；
`complement_idx([[0, 1, 2, 3]], dim=4)` → 形状 `(1, 0)`。

---

## 5. `extra_token`：加权求和，不是加权平均

### ★ 要点 5.1：`torch.sum` —— 求和

```python
extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
```

`torch.sum` 是**求和**。很多实现会写成「除以权重和」变成平均 —— 官方没有。

实测：`non_cls = [[1], [2], [3], [4]]`、`cls_attn = [0.1, 0.2, 0.3, 0.4]`、
丢弃 patch 2 和 3 → `extra_token = 0.3 * 3 + 0.4 * 4 = 2.5`
（加权平均会是 `2.5 / 0.7 = 3.5714...`）。

### ★ 要点 5.2：权重**不归一化**

权重直接用 `cls_attn` 的值（softmax 之后的小数），不做任何归一化。
所以 `extra_token` 的量级随「被丢弃 token 的注意力总和」变化。

### ★ 要点 5.3：`non_topk` 与 `non_topk_attn` 顺序一致

两者都用 `compl` 做 gather，所以**都按补集的升序**排列。
如果一个按补集顺序、另一个按别的顺序，权重就会错位。

### ★ 要点 5.4：`extra_token` 放在**末尾**

```python
x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
```

序列是 `[CLS] + [保留的 patch] + [extra_token]` —— `extra_token` 在**最后**，
不是最前。

---

## 6. `Block.forward` 的执行顺序

### ★ 要点 6.1：重组发生在**残差之后**

```python
tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
x = x + self.drop_path(tmp)          # ← 先加注意力残差

if index is not None:
    non_cls = x[:, 1:]               # ← 再从【已加残差】的 x 里取 patch
    ...
x = x + self.drop_path(self.mlp(self.norm2(x)))   # ← 最后才过 MLP
```

也就是说 gather 出来的 patch 是**加了注意力残差之后**的向量，
而不是残差之前的。手工复刻 `Block.forward` 与官方跑下来 max abs diff = 0。

### ★ 要点 6.2：MLP 在重组**之后**

`self.mlp(self.norm2(x))` 作用在**重组后**的短序列上。Token 变少之后
MLP 的计算量也一起降下来了 —— 这正是剪枝省算力的地方。

### ★ 要点 6.3：`n_tokens` 是输出长度减 1

```python
n_tokens = x.shape[1] - 1
```

减掉的是 CLS。fuse 时 `n_tokens = left_tokens + 1`（多出来的那个是
`extra_token`），不 fuse 时 `n_tokens = left_tokens`。

---

## 7. 官方源码的三个复现坑

### 坑 1：`topk` 的降序不是「顺手」

直觉上「选出 top-k」会写成 `sorted(range(n), key=lambda i: -score[i])`
然后取前 k 个 —— 这是**升序**。官方要的是 `torch.topk` 的**降序**输出。
两者在集合上相同，但**顺序不同**，而顺序直接决定重组后序列的排列。

验证手法：把分数改成 `[0.1, 0.5, 0.3, 0.05, 0.05]`，官方给 `idx = [1, 2, 0]`。

### 坑 2：`self.keep_rate` 与 `keep_rate` 的双重身份

第一次读这行代码几乎所有人都会把它当成一个条件表达式读完，
但 `self.keep_rate`（构造时）和 `keep_rate`（调用时）是**两个变量**。
构造时写 `keep_rate=1.0` 的模块，调用时传 `keep_rate=0.5` **不会**触发剪枝。

验证手法：用官方 `Attention` 类，`keep_rate=1.0` 构造，
调用时传 `keep_rate=0.5` 且 `tokens=None` → `index is None`。

### 坑 3：`extra_token` 的量级

因为权重不归一化，`extra_token` 的绝对值会随「被丢弃 token 的注意力总量」
浮动。如果你的实现输出的是一个稳定的「均值」量级，几乎肯定是多做了一次除法。

验证手法：`cls_attn` 全为 `0` 时，`extra_token` 必须是**全 0**（不是 NaN）；
权重和为 `0.7` 时 `weight_sum` 必须是 `0.7` 而不是 `1.0`。

---

## 8. 三层验证结构

本题的验证分三层，任何一层单独都不够：

1. **官方层**：直接从 upstream `evit.py` / `helpers.py` 用 AST 切片逐字抽取
   `Attention` / `Block` / `Mlp` / `DropPath` / `complement_idx`，
   只剥离 `timm` 依赖（评测环境没有 `timm`），跑官方代码本身。
2. **参考解层**：纯 Python + `Fraction` 的实现（本题的 `solution.py`）。
3. **oracle 层**：不依赖参考解、按官方算式独立重写一遍的纯 float 实现。

实测 34 个用例，三层两两 diff 均 ≤ `2.776e-17`。

**为什么要三层**：官方代码里有好几处「能跑通但语义不对」的地方 ——
比如 `topk` 的顺序、`self.keep_rate` 的双重身份，如果只对着参考解测，
一个把 `topk` 写成升序的实现可能「恰好」在某些用例上通过。
钉死语义必须靠**官方层**当基准。

---

## 9. mutation 的三条原则

本题的 mutation 清单共 37 项，最终 **37/37 CAUGHT**。踩过的坑：

1. **等价变异要主动剔除**。第一轮里有两项存活，逐一核对后确认是
   **真等价**：
   - 把 `per_head.append(hr[0][1:])` 改成「把所有行拼平再取 1..N-1」——
     拼平后前 N-1 个元素恰好还是第 0 行的 `1:`，**数值完全相同**；
   - 把 `out = [cls_row] + kept` 改成 `out = ([cls_row] + kept)` —— 纯括号。

   这类变异存活不代表判题弱，要先自查是不是等价，再决定是删掉还是换成一个
   **真的会产生不同数值**的等价写法（比如改成「所有行求和」）。

2. **校验层必须拆成互不依赖的独立防线**。每个 check 里都同时写：
   - 「结果等于 oracle」；
   - 「结果**不等于**那个看起来也合理的错误候选」。

   例如 `extra_token` 那条，既断言等于 `2.5`，也断言不等于 `2.5 / 0.7`
   （加权平均）和 `7.0`（不加权求和）。只写前者的话，一个「恰好在某个
   权重和为 1 的用例上相等」的错误实现会漏网。

3. **不可观测的差异要设计专门的用例，且锚点必须落在真正会变的输入上**。
   第一轮有三项存活，全都是**用例缺口**：
   - `tokens` 覆盖写成 `min(tokens, ceil)`：只在 `tokens > ceil` 时才可观测，
     于是补了一个 `keep_rate=0.3`（ceil = 2）+ `tokens=4` 的用例；
   - `left_tokens >= 1` 的断言被放宽成 `>= 0`：只在 `tokens=0` 时可观测，
     于是补了一个 `tokens=0` 期望抛异常的用例；
   - `kept_positions` 少加了 1：只在**非 fuse** 分支被断言到，
     于是给非 fuse 分支也补了 `kept_positions` 断言。

   锚点也要落在真正会变的地方 —— 第一轮有三个变异因为锚点缩进写错
   （16 空格 vs 实际 12 空格）而报 `ANCHOR-ERROR`，必须逐个核对真实缩进。

---

## 10. 官方源码位置索引

| 符号 | 位置 |
|---|---|
| `Attention.forward` 的选择分支 | `evit.py:207-224` |
| `cls_attn = attn[:, :, 0, 1:]` | `evit.py:215` |
| `cls_attn = cls_attn.mean(dim=1)` | `evit.py:216` |
| `torch.topk(..., largest=True, sorted=True)` | `evit.py:217` |
| `index = idx.unsqueeze(-1).expand(-1, -1, C)` | `evit.py:220` |
| `Block.forward` 的重组段 | `evit.py:250-266` |
| `x = x + self.drop_path(tmp)` | `evit.py:251` |
| `non_cls = x[:, 1:]` | `evit.py:255` |
| `x_others = torch.gather(non_cls, dim=1, index=index)` | `evit.py:256` |
| `compl = complement_idx(idx, N - 1)` | `evit.py:259` |
| `extra_token = torch.sum(...)` | `evit.py:263` |
| `torch.cat([x[:, 0:1], x_others, extra_token], dim=1)` | `evit.py:264` |
| `n_tokens = x.shape[1] - 1` | `evit.py:269` |
| `complement_idx` | `helpers.py` |
