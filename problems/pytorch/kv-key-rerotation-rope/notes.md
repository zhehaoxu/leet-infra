# 解题说明 · KeyRerotation

## 问题：驱逐破坏了 RoPE 的"相对位置"

RoPE 的精髓是：两个位置 $p, q$ 的 key/query 做点积时，结果只依赖**相对距离** $q-p$（旋转角相减）。KV 压缩把位置 $0..L-1$ 里的一批 token 挑出来，**在缓存里重新排成 $0..n\_kept-1$**。如果直接 gather 就走，保留 key 携带的还是"原位置角"，而 query 用的是"新位置角"——两者相减不再是真实的相对距离，注意力打分就错位了。

## 解法：把角度差补回来

设原位置 $p$ 的 key 已经过 $R(p)$ 旋转。想让它表现得像在**新位置 $i$**，需要再乘一个 $R(i-p)$：

$$k' = R(\delta)\,k,\quad \delta = i - p$$

展签到逐维旋转就是题面公式：`out = k*cos(emb) + rotate_half(k)*sin(emb)`，`emb = [δ·inv_freq, δ·inv_freq]`。这与标准 RoPE 前向同构，只是把"位置"换成了"位置差"。

## rotate_half 的方向约定

不同仓库 `rotate_half` 定义可能差一个符号（HF 是 `[-x2, x1]`）。本题把定义写死在题面，oracle 与实现都用同一条，只要按题面走就对得上。

## delta 为负没问题

保留的位置总是 $\ge$ 新位置（因为丢过中间 token），所以 $\delta=i-p\le 0$。$\cos$ 是偶函数、$\sin$ 是奇函数，负角度自动等价于"反向旋转"，不需要特判。

## 两个恒等自检

- **连续保留** $[0,1,...,L-1]$：$\delta\equiv 0$ → 输出应完全等于原 `keys`（`check_identity_when_contiguous`）。
- **位置 0 留在新位 0**：该行 $\delta=0$ → 原样不变。

这两条是排查符号/拼接写反的最快断言。

## 与 ChunkPress/ChunkKV 的关系

KeyRerotation 本身**不决定留谁**，它是个"后处理包装器"：任何 ScorerPress 选出 indices 后，都可以套一层重旋转让 RoPE 保持自洽。本题把这层单独拿出来，输入直接给 indices。
