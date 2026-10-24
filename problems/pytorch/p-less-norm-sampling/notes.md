# 解题说明 · P-less-norm 采样

## 与 P-less 的唯一区别：阈值归一

原版 P-less 直接令 `thr = Σp²`。P-less-norm 做了一次仿射变换：

$$\text{thr}_{\text{norm}} = \frac{v\,\Sigma p^2 - 1}{v-1}$$

把 $\Sigma p^2$ 的定义域 $[1/v,\,1]$ 线性映射到 $[0,\,1]$：

| 分布形状 | $\Sigma p^2$ | P-less thr | P-less-norm thr |
|----------|-------------|-----------|-----------------|
| 均匀 ($1/v$) | $1/v$ | $1/v$ | $0$ |
| 尖峰 ($p_{\max}\to1$) | $\to 1$ | $\to 1$ | $\to 1$ |

两者在尖峰端重合，但在中段 **norm 阈值更低** → 更多 token 满足 $p_i\ge\text{thr}$ →
保留集合更大、采样更发散。这正是论文里「relaxed from p-less，favor diversity」的含义。

## 合法性（不会全被清空）

需证明 $p_{\max}\ge\text{thr}_{\text{norm}}$ 恒成立：由 $\Sigma p^2\le p_{\max}$，

$$\text{thr}_{\text{norm}} = \frac{v\,\Sigma p^2 - 1}{v-1} \le \frac{v\,p_{\max}-1}{v-1} \le p_{\max}$$

第二个不等式等价于 $v\,p_{\max}-1 \le (v-1)p_{\max} \iff p_{\max}\le 1$，恒成立。故 argmax
永远满足阈值，`filtered` 分母恒正。

## 实现要点

`v = probs.shape[-1]`；`thr = (v*probs.pow(2).sum(-1,keepdim=True) - 1)/(v - 1)`；
`keep = probs >= thr`；归一化后对 `filtered` 原顺序 CDF 取首个 $\ge u$。官方实现用
`mask = probs < p` 剔除**严格小于**者，等价于保留 `probs >= thr`（本题用 `>=`）。
注意 `V ≥ 2`，否则分母为 0。
