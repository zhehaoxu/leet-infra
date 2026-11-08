# EViT token 重组织账

## 背景

视觉 Transformer 里有一个很朴素的省钱思路：**别把全部 patch token 都带进下一层**。
但「丢掉哪些」这件事不能拍脑袋——如果随手丢，模型就看不到那些区域了。

EViT（*Early Vision Transformer*）给了一个很干净的做法：**让 CLS token 自己去投票**。

它的核心观察是：CLS token 作为 query 时算出的那一行注意力，天然就是「全局哪里重要」
的一个读数。于是 EViT 做了三件事：

1. **打分**：取 attention 的第 0 行（CLS 作为 query），去掉 CLS 自己那一列，
   跨所有 head 求平均，得到每个 patch 的一个重要性分数；
2. **筛选**：按这个分数取 top-k，保留分数最高的那些 patch；
3. **重组**：把保留下来的 patch 重新排好序，再把所有被丢弃的 patch
   **加权求和**聚成**一个** `extra_token` 追加到序列末尾。

第 3 步是关键设计：被丢弃的 token 不是直接扔掉，而是把它的信息「压」成一个 token
带下去，这样后续层仍然能感知到「那里曾经有过东西」。

对应的官方源码在 `youweiliang/evit`（默认分支 `master`）：

| 位置 | 作用 |
|---|---|
| `evit.py::Attention.forward` | cls_attn 打分 + topk 选 token |
| `evit.py::Block.forward` | token 重组（gather / fuse） |
| `helpers.py::complement_idx` | 补集计算 |

本题要你用纯 Python 复现这套规则（三个函数）的**精确数值口径**。
这条链路上有至少八处「看起来合理但官方不是这么写」的细节，
本题的判题会把它们逐个钉死。

## 口径约定

所有 mode 的入参都是**纯 Python 列表**，不依赖任何张量形状推断：

- `attn`：**`[H][N][N]`** —— H 个 head，每个 head 是一个 N×N 的注意力矩阵。
  N 含 CLS token，即 `N = patch 数 + 1`。每个 head 的每一行都应是非负的注意力分布。
- `hidden`：**`[N][C]`** —— N 个 token，C 个通道，**第 0 行是 CLS token**。
- `non_cls`：**`[N-1][C]`** —— 已经去掉 CLS 之后的 patch 序列，
  下标 `i` 对应原序列的第 `i+1` 个 token。
- `cls_attn`：长度 `N-1` 的打分行，下标 `i` 对应第 `i` 个 patch。
- `idx`：被保留的 patch 在 **patch 空间**（0 基，不含 CLS）里的下标。
- `complement`：`idx` 在 patch 空间里的补集。
- 所有出参统一为 `float`（内部能走精确有理数的部分走 `fractions.Fraction`）。
- 非法输入一律抛 `ValueError`。

---

## mode 1：`cls_attn_score`

复现官方打分口径：

```python
cls_attn = attn[:, :, 0, 1:]   # [B, H, N-1]
cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
```

**签名**

```python
cls_attn_score(attn)
    → {"cls_attn": [...], "n_heads": int, "n_tokens": int}
```

**要点**

1. ★ 只取 attention 的**第 0 行** —— 也就是 **CLS 作为 query** 的那一行。
   其余行（patch 作为 query）**完全不参与**打分。
2. ★ 列从 **1** 开始 —— **去掉 CLS 自身那一列**（CLS 看自己的注意力权重没有意义）。
   所以输出长度是 `N-1`，不是 `N`。
3. ★ 跨 head 求**算术平均**（`mean`），不是求和、不是取最大。
4. `n_heads` = H，`n_tokens` = N。

**示例**

```python
attn = [
    [[0.5, 0.1, 0.2, 0.1, 0.1],   # head 0，CLS 那一行
     [0.1, 0.4, 0.2, 0.2, 0.1],
     [0.1, 0.1, 0.4, 0.2, 0.2],
     [0.1, 0.1, 0.2, 0.4, 0.2],
     [0.1, 0.2, 0.1, 0.2, 0.4]],
    [[0.2, 0.3, 0.2, 0.2, 0.1],   # head 1，CLS 那一行
     [0.2, 0.2, 0.3, 0.2, 0.1],
     [0.1, 0.2, 0.2, 0.3, 0.2],
     [0.2, 0.1, 0.2, 0.2, 0.3],
     [0.3, 0.1, 0.2, 0.1, 0.3]],
]
solve("cls_attn_score", attn=attn)["cls_attn"]
# → [(0.2 + 0.3) / 2, (0.2 + 0.2) / 2, (0.1 + 0.2) / 2, (0.1 + 0.1) / 2]
#   = [0.25, 0.2, 0.15, 0.1]
```

---

## mode 2：`select_tokens`

复现官方的 token 数量决策与 topk 选择：

```python
left_tokens = N - 1
if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
    left_tokens = math.ceil(keep_rate * (N - 1))
    if tokens is not None:
        left_tokens = tokens
    if left_tokens == N - 1:
        return x, None, None, None, left_tokens
    assert left_tokens >= 1
    cls_attn = attn[:, :, 0, 1:]
    cls_attn = cls_attn.mean(dim=1)
    _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)
    index = idx.unsqueeze(-1).expand(-1, -1, C)
    return x, index, idx, cls_attn, left_tokens
return x, None, None, None, left_tokens
```

**签名**

```python
select_tokens(cls_attn, keep_rate, self_keep_rate=1.0, tokens=None)
    → {"left_tokens": int, "triggered": bool, "idx": [...] | None,
       "scores": [...] | None, "kept_scores": [...] | None,
       "complement": [...] | None, "n_tokens": int}
```

**要点**

1. ★ **触发条件里的两个变量是不同的东西**：
   - `self_keep_rate` 是**构造时**写进 `Attention` 的 `self.keep_rate`；
   - `keep_rate` 是**调用时**传进来的参数。

   官方那行 `self.keep_rate < 1 and keep_rate < 1 or tokens is not None`
   因为 Python 里 `and` 优先级高于 `or`，等价于
   `(self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)`。
   所以：**两个比率必须都 `< 1` 才走 keep_rate 分支**；
   `tokens` 则可以**单独**触发。
2. ★ `left_tokens = math.ceil(keep_rate * (N - 1))` —— **向上取整**，
   而且基数是 **`N - 1`**（N 含 CLS token），不是 N。
3. ★ `tokens` 不为 `None` 时，`left_tokens` 被**直接覆盖**为 `tokens`
   （keep_rate 算出的 ceil 结果被丢弃）。覆盖是**直接赋值**，
   不是取 `min` —— `tokens` 比 ceil 结果大时也照用。
4. ★ `left_tokens == N - 1` 时**提前返回**：不选 token，
   `idx` / `scores` / `kept_scores` / `complement` **全为 `None`**。
   注意此时**条件是触发了的**（`triggered` 仍为 `True`），
   只是因为「要保留的正好是全部」而没必要选。
5. `left_tokens` 必须 `>= 1`，否则官方 `assert` 失败 → 你的实现应抛异常。
6. ★ `torch.topk(..., largest=True, sorted=True)` 的返回值**按分数降序**排列，
   **不是**按原下标升序。分数并列时 torch 按下标升序取。
7. `complement` 是 `idx` 在 patch 空间（长度 `N-1`）里的补集，
   按**下标升序**（官方 `complement_idx` 内部 `torch.sort(descending=False)`）。
8. `scores` 与 `kept_scores` 是被选中 token 的分数（按 `idx` 的顺序）；
   提前返回时为 `None`。
9. `n_tokens` 恒为 `N`（= patch 数 + 1）。

---

## mode 3：`fuse_token`

复现官方把被丢弃 token 压成一个 `extra_token` 的那一步：

```python
compl = complement_idx(idx, N - 1)                              # [B, N-1-left_tokens]
non_topk = torch.gather(non_cls, dim=1,
                         index=compl.unsqueeze(-1).expand(-1, -1, C))
non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1),
                        dim=1, keepdim=True)                    # [B, 1, C]
```

**签名**

```python
fuse_token(non_cls, cls_attn, idx, complement)
    → {"extra_token": [...], "non_topk": [[...]], "non_topk_attn": [...],
       "weight_sum": float}
```

**要点**

1. ★ `extra_token` 是**加权求和（sum）**，**不是加权平均**：
   `sum_i(w_i * x_i)`。最常见的错法是除以权重和变成平均。
2. ★ 权重直接用 `cls_attn` 的值（softmax 之后的小数），**不做归一化**。
   所以 `extra_token` 的量级随权重和变化，不是一个「均值」。
3. `non_topk` 与 `non_topk_attn` 都按 `complement` 的顺序排列
   （`complement` 本身是下标升序的）。
4. `weight_sum` 是被丢弃 token 的权重之和（**未归一化**），
   用作「这里确实做了加权求和而非平均」的证据。
5. `idx` 本身不参与数值计算（补集已由 `complement` 给出），
   但仍需与 `non_cls` / `cls_attn` 的长度自洽。

**示例**

```python
non_cls = [[1.0], [2.0], [3.0], [4.0]]
cls_attn = [0.1, 0.2, 0.3, 0.4]
complement = [2, 3]     # 丢弃 patch 2 和 patch 3
solve("fuse_token", non_cls=non_cls, cls_attn=cls_attn,
      idx=[0, 1], complement=complement)["extra_token"]
# → [0.3 * 3.0 + 0.4 * 4.0] = [2.5]
#   （不是 (0.3*3 + 0.4*4) / (0.3+0.4) = 3.5714...）
```

---

## mode 4：`reorganize`

复现官方 `Block.forward` 里的重组：

```python
non_cls = x[:, 1:]
x_others = torch.gather(non_cls, dim=1, index=index)   # [B, left_tokens, C]

if self.fuse_token:
    compl = complement_idx(idx, N - 1)
    non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
    x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
else:
    x = torch.cat([x[:, 0:1], x_others], dim=1)
```

**签名**

```python
reorganize(hidden, idx, extra_token=None, fuse=True)
    → {"out": [[...]], "n_tokens": int, "kept_positions": [...],
       "extra_token": [...] | None}
```

**要点**

1. ★ 输出序列是 **`[CLS] + [按 idx 顺序 gather 的 patch] + ([extra_token] 若 fuse)`**。
2. ★ `extra_token` 放在**末尾**，不是最前。把它插到 CLS 后面是最常见的错法。
3. ★ gather 出来的 patch 保持 `idx` 的顺序（**分数降序**），
   **不是**原下标升序。
4. `hidden` 是**已经加过注意力残差**的序列 —— 官方在 `x = x + drop_path(tmp)`
   之后才做重组。
5. 输出长度：fuse 时 = `1 + len(idx) + 1`；不 fuse 时 = `1 + len(idx)`。
6. `n_tokens` = 输出长度 - 1（去掉 CLS）。
7. `kept_positions` 是被保留 patch 在**原序列**里的位置
   （patch 下标 `i` → 原序列第 `i+1` 行，因为第 0 行是 CLS）。
8. 不 fuse 时 `extra_token` 出参为 `None`。

---

## 常见错误清单

1. `cls_attn` 对**所有行**求均值，而不是只取第 0 行（CLS 作为 query 的那行）。
2. `cls_attn` 把 **CLS 自己那一列**也算进去（输出长度成了 N）。
3. `cls_attn` 跨 head 求**和**或取**最大**，而不是算术平均。
4. 触发条件写成 `keep_rate < 1` —— 忽略了**构造时**的 `self.keep_rate`。
5. 触发条件写成 `self_keep_rate < 1 and keep_rate < 1` —— 丢掉了
   `tokens` 可以**单独触发**这条路径。
6. `left_tokens` 用 **floor** 或 **round** 而不是 `ceil`。
7. `left_tokens` 的基数用 **N** 而不是 **N - 1**。
8. `tokens` 覆盖写成 `min(tokens, left_tokens)` 而不是**直接赋值**。
9. `left_tokens == N - 1` 时没有**提前返回**（仍然去选 topk）。
10. 提前返回时 `idx` 返回了空列表而不是 `None`。
11. topk 结果**按下标升序**返回，而不是按分数降序。
12. `complement` 按下标**降序**返回（官方是升序）。
13. `extra_token` 算成**加权平均**（除了权重和）。
14. `extra_token` 的权重被**归一化**了。
15. `extra_token` 忽略了权重（直接对 patch 求和）。
16. `extra_token` 放在序列**最前**（CLS 之后），而不是末尾。
17. `reorganize` 把 CLS 也算进 patch 序列（`hidden[1:]` 写成 `hidden`）。
18. `reorganize` 漏掉了 CLS 那一行。
19. `n_tokens` 算成输出长度（忘了减 1 去掉 CLS）。
20. `kept_positions` 返回 patch 空间下标，而不是原序列位置。

## 自检

```bash
python starter.py            # 骨架自检
```

提交前请确认：

- 四个 mode 的返回键名与题面完全一致。
- 所有出参是 `float`（不是 `Fraction`）。
- 非法输入抛 `ValueError`。
- `select_tokens` 在提前返回时，`idx` / `scores` / `kept_scores` /
  `complement` 都是 `None`。
