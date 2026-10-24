# 解题说明 · DRY 采样

## 它在罚什么

DRY 的目标是打断 **n-gram 循环**。给定上下文末尾的 $N$ 个 token（`last`），我们想知道：
「把最后若干个 token 当作模式，它在更早的位置重复出现了几次？」若末尾的 `a b c` 之前已经
原样出现过，那么再吐出 `a b c` 之后的那一个 token 就会**延续**这个循环，于是惩罚那个
「下一个 token」。惩罚随重复长度**指数**增长，够长的循环会被罚到几乎不可能再出现。

## 反向 Z-algorithm

llama.cpp 把 `last` 逆序（`rat(j)` = 从末尾数第 $j$ 个），再跑 **Z-algorithm**：
$z[k]$ = `rev` 与 `rev[k:]` 的最长公共前缀。这正是「以末尾为锚、向后偏移 $k$ 后仍相同的前缀长」
——即一段从当前位置能往回对齐的重复。Z-algorithm 用一对游标 `[lt, rt]`（当前 Z-box）把
朴素 $O(N^2)$ 降到 $O(N)$：

```
for k in 1..N-1:
    if k > rt:                 # 落在盒外，朴素扩展
        n = 0; while k+n<N and rev[n]==rev[k+n]: n+=1
        z[k] = n; if n>0: lt,rt = k, k+n-1
    else:                      # 落在盒内，利用已算的 z[k-lt]
        p = k - lt; right = rt - k + 1
        if z[p] < right: z[k] = z[p]
        else: 从 rt+1 继续扩展 → z[k] = i-k; lt,rt = k, i-1
```

本题允许朴素实现（每个 $k$ 各自 while 比对），判题用**独立的暴力最长公共前缀**做 oracle
对拍，两条路径必须逐值一致。

## 惩罚 token 的换算（易错点）

C++ 里 step3 用 `rat(last_n_repeat - 2 - i)`，其中 `dry_repeat_count[i]` 存的是偏移
$k = (N-1) - i$ 的 $z$ 值。代入化简后，命中 shift $k$ 时被罚的 token 恰为**时间正序**的
`last[N - k]`（$k = 1..N-1$）。务必不要写成 `last[N-1-k]` 之类的 off-by-one：那会罚错一个
位置、整题挂掉。用两个手算示例反复核对（见题面）。

- `history=[5,6,7,5,6,7]`：只有 $k=3$ 的 $L_3=3$ 达标 → `last[6-3]=last[3]=5`。
- `history=[0,1,0,1,0,1]`：$k=2$ 的 $L_2=4$ 达标 → `last[6-2]=last[4]=0`。

## 其它细节

- 同一 token 被多个 shift 命中：取**最大**重复长（幂最高、罚最重）。
- `max_exponent` 防 `base**exp` 浮点溢出：仅当 `base > 1.000001` 才启用，正常示例用不到。
- 早退：`mult==0`、`base<1`、`last_n==0`、$N\le$ `allowed` 四种情况直接原样返回。
