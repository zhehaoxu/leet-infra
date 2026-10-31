# 参考做法：一次同步取 offs，循环 E 组，每组三段切片矩阵乘

## 实现

1. 校验照 `_grouped_mm_fallback_fake` 的断言清单来，再补上题面独有的三条：`offs` 非降、元素非负、末值不超过行数。形状检查要在 `torch.mm` 之前做完，否则「`K != N` 时梯度方向写反」这种错误只会以 `RuntimeError` 的形式泄漏出去。
2. `out`、`grad_input`、`grad_weight` 全部 `torch.zeros` 起手，然后**只覆盖有主人的行**。尾部 $[\mathit{offs}[E-1],S)$ 和空组对应的 `grad_weight[e]` 就这样天然留 0——不要为它们写特判。
3. `offs.tolist()` 在循环外做一次，循环里只用 host 侧的整数。`counts` 由差分给出：$\mathit{count}_e=\mathit{offs}[e]-\mathit{offs}[e-1]$（$e=0$ 时左端记 0）。
4. 每组三段 `torch.mm`，都用 `out=` 写进预分配好的切片：前向 `x @ w`、输入梯度 `g @ w.T`、权重梯度 `x.T @ g`。后两段的**左矩阵不同**，这是本题最容易写反的一处。
5. 空组 `if start == end: continue`，但 `start = end` 照常推进——其实两者都是空操作，写清楚只为让读的人确认区间划分没停摆。

四个常见错误：

- **把 `offs` 当组大小**（`end = start + offs[e]`）或者**当区间起点**（`table = offs + [S]`，第 $e$ 组取 `table[e:2]`）。前者只有第一组是对的，后者整条分派往左挪一格、最后一组还会顺手把尾部行吃掉。
- **`grad_weight` 写成 `g.T @ x`**：得到的是 $[N,K]$。$K\neq N$ 时形状检查就能挡住；$K=N$ 时形状完全合法、数值却被转置了，所以判题里 $K=N$ 与 $K\neq N$ 的用例都有。
- **用 `torch.empty` 起手**：见下面的变异测试结论，这条**判题抓不稳**。
- **忘了 `weight[e]` 的下标就是组的下标**：某些实现按「非空组」重新编号，于是中间有空组时后面全部错位。判题用「把空专家的权重换成随机值」专门钉这一点。

## 正确性检验

8 个 `check_*`：手算示例（四项返回值全部核对，含空组与尾部行）、随机对拍（8 组形状/分配/量级组合，含 `float64` 随机数一组，`offs` 换 int32、`x`/`weight`/`grad_output` 各造一种非连续视图）、分组代数不变量（组内行置换必须让 `out`/`grad_input` 同步置换而 `grad_weight`/`counts` 逐位不动；**空专家的权重换成随机值后四个返回项完全不变**；`counts` 与 `offs` 差分一致、和恰为 $\mathit{offs}[E-1]$）、全空与全占（`offs` 全 0、只有首组非空、只有末组非空、中间两个空、单专家、int32）、尾部行（三种 `offs` 下尾部必须全 0，并显式核对一段小数据的四项数值）、精确性（0.25 倍数的数据在 `rtol=atol=0` 下与纯 Python 标量三重循环逐位相同）、非法输入、最大形状 $32\times16$。

oracle 是纯 Python：逐组、逐行、逐列的标量三重循环，不调用任何 `torch` 矩阵乘，也不参考解里的切片写法。整数与 0.25 倍数都是二进制精确的，所以「逐位相同」这条不会靠容差蒙过去。

变异测试 16 个 mutant，抓到 14 个：`offs` 当组大小、`offs` 当起点、末组吞掉尾部、`counts` 用 `offs` 本身、`counts` 差一（把右端点算进去）、输入梯度不转置、输入梯度错用 `x`、权重梯度写成 `x @ g.T`（形状仍是 $[K,N]$ 但数值彻底错位）、权重梯度写成 `g.T @ x`、权重梯度算对再整体取转置、反向误用前缀 $[0,\mathit{end})$、所有组共用 `weight[0]`、单行组被跳过、`torch.empty` 起手。两个**判不出来**：

- **空组「不跳过而是拿下一个专家的权重乘一下」**——`x[start:start]` 是 0 行，`torch.mm` 写 0 行，任何权重都不参与运算。这是**真等价**，题面已经把话说清：能被判出来的是「非空组拿错 `weight` 下标」（`weight_shared` 这个 mutant 就是它的一个变体，被抓到了），而不是空组自不自乘。
- **`torch.empty` 起手**——16 项里只有 `check_maximum_shape` 那次失败，因为本机小尺寸分配拿到的恰好是零页。我在 `check_all_empty_and_full_offsets` 和 `check_tail_rows` 里都加了「先用一次全分派把同形状缓冲填上非零值再丢弃」的动作，也没能稳定把它暴露出来。结论要如实说：**这条口径不能靠判题保证**，它是题面写死、靠实现者自己守的规则；判题只在内存布局碰巧脏时才会报错。

## 范围与来源

切分口径、空组跳过、尾部行留 0、前反向的三段写法，全部取自本仓库判题环境所用 transformers 5.20.2 的 `integrations/moe.py`——它就是 grouped-mm 不可用时（非 bf16 权重、`torch.compile` fullgraph、没有对应 kernel）实际会跑到的那条 fallback 路径，`_grouped_mm_fallback_fake` 的断言清单也被我原样搬进校验（只额外加了 `offs` 单调与不越界两条）。逐字摘录：

```text
# transformers/integrations/moe.py
def _grouped_mm_fallback(input, weight, offs):
    output = torch.zeros(input.size(0), weight.size(2), ...)
    start = 0
    # single cpu<->gpu sync point here,
    # avoids multiple syncs inside the loop
    for i, end in enumerate(offs.tolist()):
        if start == end:
            continue
        torch.mm(input[start:end], weight[i], out=output[start:end])
        start = end
    return output

def _grouped_mm_fallback_fake(input, weight, offs):
    assert input.dim() == 2
    assert weight.dim() == 3
    assert offs.dim() == 1
    assert offs.size(0) == weight.size(0)
    assert input.size(1) == weight.size(1)
    assert offs.dtype in (torch.int32, torch.int64)

def _grouped_mm_fallback_backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)
    start = 0
    for i, end in enumerate(ctx.offs.tolist()):
        if start == end:
            continue
        torch.mm(grad_output[start:end], weight[i].T, out=grad_input[start:end])
        torch.mm(input[start:end].T, grad_output[start:end], out=grad_weight[i])
        start = end
    return grad_input, grad_weight, None
```

两处**题面自定**、不做归属声明：`counts` 这个返回值和它的 int64 口径（上游没有这个输出）；非法输入一律转成 `ValueError`（上游是 `assert`）。上游把这三段挂在 `torch.library.custom_op` 的 forward/backward 上、由 autograd 触发，本题把它拉平成纯函数——不建计算图、不调 `backward()`，梯度当成普通矩阵乘来算。

与合集内其他题的分工：`moe-grouped-execution-plan` 管**怎么排**这些组（把 ragged 组塞进定长 tile 的执行计划、padding 与调度代价），本题管**排好之后怎么算**；把 token 搬到 `x` 的这个顺序上去是 `moe-all-to-all-dispatch` 与容量分派两件事（`moe-capacity-dispatch`、`moe-capacity-jitter-dispatch`）的职责；专家权重的语义（共享专家、上下投影）见 `moe-shared-expert-forward`。
