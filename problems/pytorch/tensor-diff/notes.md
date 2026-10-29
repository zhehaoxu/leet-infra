# 详解：diff（n 阶差分与 prepend/append）

## 出处

`torch.diff` 自 1.5 起提供，语义照 numpy 的 `np.diff`（`prepend`/`append` 两个参数就是 numpy 的 `np.diff(..., prepend=, append=)` 接口）。使用场景极多：把位置 id 序列转成步长（RoPE 外推里检查 `position_ids` 是否连续）、把 token 区间转成边界、时间序列去趋势、有限差分做梯度检查（与 #65 的 gradcheck 同源，只是那里比较的是 $f(x+\epsilon)-f(x-\epsilon)$ 与解析梯度），以及「二阶差分为负即非凸」这类判据。

## 差分算子与它的两个细节

沿轴 $d$ 的一阶差分是一个 $(L-1)\times L$ 的带状矩阵作用在该轴上：

$$(\Delta x)_i = x_{i+1} - x_i, \qquad \Delta^{(n)} = \underbrace{\Delta^{(1)} \circ \cdots \circ \Delta^{(1)}}_{n}$$

实现上只有两个坑：

1. **任意轴**。切片写法 `x[:, 1:] - x[:, :-1]` 把轴名硬编码进代码，换 `dim=1` 就得改一遍。`torch.narrow` 把「取第 $i$ 根轴上从 $s$ 开始的 $l$ 个元素」抽象成通用操作，于是

   ```
   y = y.narrow(axis, 1, size - 1) - y.narrow(axis, 0, size - 1)
   ```

   一行对任何轴成立，且 `size` 每轮重新读取（因为长度逐轮 −1）。
2. **边界先拼接再差分**。`prepend` 的**最后一行**与 `x` 的第一行相邻，`append` 的第一行与 `x` 的最后一行相邻，所以顺序必须是 `cat([prepend, x, append], dim=axis)`；输出长度

   $$L_{\text{out}} = \dim_x + \dim_p + \dim_a - n$$

   `prepend`/`append` 为 None 时按 0 计。把 `n` 只减一次的实现会在 $n\ge 2$ 时把长度算长。

$n$ 阶差分等价于「先把 $n$ 次一阶差分连做」，判题用恒等式 `diff(x, n=2) == diff(diff(x), n=1)` 直接验证这条递归定义。

## 与累计和的关系

差分是 `cumsum` 的左逆：$\Delta\big(\mathrm{cumsum}(x)\big)_i = x_{i+1}$（丢掉首元素），$\mathrm{cumsum}(\Delta x) = x - x_0$。因此「给差分还原原序列」必须额外给初值或 `prepend`，这解释了 numpy/torch 为什么要留这两个参数——没有边界值，差分是不可逆的。

## 核心与易错点

1. **`n=0` 要返回副本**：判题断言 `torch.allclose(solve(x, n=0), x)`，同时语义上不应把入参对象交回去（改了输出会影响输入）。
2. **长度非正报错**：`x` 沿该轴长 2、`n=3` 时 torch 给空张量，本题要求 ValueError——实现要在每轮 `narrow` 前检查 `size < 2`。
3. **`prepend`/`append` 的校验粒度**：ndim 必须相同、除 `dim` 外每一维必须相同、dtype 必须相同；只比 shape 元组长度会漏掉「第 1 维不符」这种错。
4. **bool 不是 int**：`n=True` 会溜进 `n==1` 分支，参考答案显式排除。
5. **差分不改变 dtype**：全程 float32 即可（无稳定化需求），这一点与本题族其他成员不同。

## 与邻近题的区别

- `reduce-axis` / `tensor-shape-indexing`：归约与形状操作，没有「相邻两个元素做差」的语义。
- `concat-stack-split`：拼接是本题处理 `prepend`/`append` 的手段，那题不考差分。
- `repeat_interleave`（order364）：同一批的「索引重构」姊妹题——那题造下标做 `index_select`，本题造视图做 `narrow` 相减。

## 对拍口径与复杂度

- oracle 是 `torch.diff`，覆盖 $n$ 的六档、`dim` 的正负六写法、`prepend`/`append` 的单双组合，另有长度公式、二阶=一阶之差分、线性序列一阶恒 1 二阶全 0、中间轴逐行核对，以及 8 组非法入参。
- 复杂度 $O(nN)$（$N$ 为元素数），每轮一次逐元素减；无 float64 需求，判题微秒到毫秒级。
