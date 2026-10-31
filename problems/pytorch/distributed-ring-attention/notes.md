# 参考思路：把环、在线 softmax 和通信账叠在同一张时间表上

## 实现

五段结构：校验入参 → 逐 rank 重放 C 步 → 每步决定跳不跳 → 每行做在线 softmax 递推 → 收尾归一化。
把 `traces`、`sources`、`skipped`、`comm_bytes` 一起交出来，是为了让「读错一步」在判题里必然暴露：
光看 `out` 的话，把环转错方向有时也能凑出同一个加权和。

1. **环的走向**。第 $s$ 步 rank $r$ 收到 $\text{src} = (r - s) \bmod C$。第 0 步是自己的块，
   之后每步往前一个 rank 走一格，第 $s$ 步拿到的就是"上游 $s$ 跳之前"手里的那块，$C$ 步正好
   每个块见一次。
2. **整块跳过**。`causal` 且 $\text{src} > r$ 时该块的 key 全局位置全部 $> rS + S - 1$，整块不可见：
   不进 `if not dead` 分支，一个 exp 都不做，`m`、`l`、`acc` 原样留着。
3. **在线递推**。`corr = exp(m - m')`，`l` 与 `acc` 先乘 `corr` 再加新块。首块 $m = -\infty$，
   `exp(-inf - finite)` 在 Python 里是 `0.0`，正好把"还没累加过"这件事写成一个乘零，不需要分支。
4. **分母不会为零**。$m'$ 必是本块里某个 logit，那一项的 $\exp(0) = 1$ 一定被加进 `l`，
   所以每一步结束后 $l \ge 1$，末态直接 `acc / l` 即可。
5. **通信账**。每步搬一块 KV 进来、搬一块出去，计 $2SD\cdot\text{dtype\_bytes}$；最后一步之后
   环上没有下一跳，记 0。跳过只省算力：被吃掉的块照样收、照样转发。

两个等价的写法值得记一笔，它们决定了题面能写多"死"：

- $\text{src} > r \iff s > r$（在 $\text{src} = (r-s)\bmod C$ 下恒成立）：$s \le r$ 时
  $\text{src} = r - s \le r$；$s > r$ 时 $\text{src} = C - (s - r) \ge r + 1$。所以跳过判据写成
  "源下标越界"还是"步号越界"输出完全相同，题面只钉输出、不钉写法。
- `corr = exp(m - m')` **不能**写成 `1 / exp(m' - m)`：首块 $m = -\infty$ 时 $m' - m = +\infty$，
  而 `math.exp(inf)` 在 CPython 里抛 `OverflowError`，不是返回 `inf`。这不是等价改写，是真 bug，
  变异测试里它被 `check_large_logits` 抓住了。

## 正确性检验

十三条 `check_*`，主对拍用的 oracle 结构独立：先把所有分片拼成一张全局 $(N\times D)$ 矩阵，
输出走**离线 softmax**（先求全行最大值再一次性求和，完全没有 `corr`）；`traces` 每一步都把
"到目前为止可见的全部 key"**从头重算**；`sources` 由
`range(r, -1, -1) + range(C-1, r, -1)` 拼出；`skipped` 用位置比较
$\text{src}\cdot S > rS + S - 1$；`comm` 用尾部补零的列表。参考解则是逐 rank、逐步的在线递推，
两条路径不共用任何中间量。

- `check_example`：题面四个例子逐位比对（含 `traces` 与被跳过步的照抄）。
- `check_matches_oracle`：18 组 fixture（单 rank 到 8 rank、$S$ 从 1 到 6、$D$ 从 1 到 3、因果与非因果、
  `scale = 1/sqrt(D)`）加 3 组 $\times$ 3 种 `dtype_bytes` 的随机对拍。
- `check_matches_dense_attention`：另一条独立路径——用 `torch` 拼全局分数矩阵、
  `masked_fill(-inf)`、`softmax`、再乘 $V$，与环上结果按 $10^{-12}$ 相对容差比。
- `check_sources_rotate_backwards`：`sources[r][0] == r`、相邻两步差 $-1 \bmod C$、
  每行是一个排列；$C=1$ 时 `sources == [[0]]`。
- `check_causal_skip_is_whole_block`：`skipped[r][s]` 必须等于位置判据；非因果时一块都不跳。
- `check_uniform_logits_count_visible_keys`：$q$ 全零时 `l` 必须等于可见 key 数（因果是 $p+1$，
  非因果是 $N$），输出是可见 $v$ 的运行均值——这一条把"掩码只认全局位置"钉死。
- `check_first_visible_key_survives`：因果时全局第 0 行只有一个可见 key，输出必须**逐位**等于
  $v_0$（权重恰为 1，没有任何浮点噪声的余地）。
- `check_skipped_step_keeps_state`：被跳过的步 `traces` 与上一步逐位相同、且仍然计通信；
  $C=4$ 时被吃掉的块数必须是 $0+1+2+3=6$；因果与非因果的输出必须不同。
- `check_comm_account`：四种 $(C,S,D,\text{dtype})$ 下账本等于"尾部补零的 payload 列表"。
- `check_large_logits`：$q=20$、$k=20$ 给出 logit 400（不扣最大值直接 exp 就溢出），
  $q=-20$ 给出 $-400$（要求 $m$ 允许是负数），两种都要算对；另有两组 $\pm50$、`scale=30` 的
  随机用例要求全程有限。
- `check_single_rank`：$C=1$ 退化为普通因果注意力，并与手写的下三角 softmax 对拍。
- `check_scale`：8 rank $\times$ 8 行 $\times$ 4 维、单 rank 16 行 $\times$ 8 维、
  4 rank $\times$ 16 行 $\times$ 8 维，以及全零输入的 16 行阶梯。
- `check_invalid`：39 个非法入参 + 4 个合法边界。分片结构、元素类型、三个标量各一组；
  `naive` 的坑（`bool` 既是 `int` 又是"合法的 0/1"）在 `causal`、`dtype_bytes`、元素三处各挡一次。

变异测试跑了 36 个 mutant 全部被抓，另有两个等价改写不计入。这里有一段值得单说：把
`dead = causal and source > rank` 换成 `dead = causal and step > rank` 后输出**逐位相同**，
因为两条判据在上面那个恒等式下完全等价——这类"怎么改都不影响输出"的写法不该判错，所以
题面钉的是输出而不是判据形式。反过来，`corr = 1 / exp(m' - m)` 看起来只是移了个项，实际在
首块就 `OverflowError`，属于必须被抓住的真错误。

整块跳过的价值可以直接数出来（`_xcheck554.py` 的第一张表，$S=8$、$D=4$、`dtype=4`）：

```text
# C ranks, shard length 8, head width 4, causal=True, dtype_bytes=4
# tile pairs = score entries an implementation would touch if every received block were scored
   C  blocks  skipped  share | score-tile pairs skip / no-skip   saved | exp calls | comm bytes
   2       4        1   0.25 |         192 /       256    0.25 |       160 |        512
   4      16        6   0.38 |         640 /      1024    0.38 |       608 |       3072
   8      64       28   0.44 |        2304 /      4096    0.44 |      2368 |      14336
  16     256      120   0.47 |        8704 /     16384    0.47 |      9344 |      61440
```

被吃掉的块数恒为 $C(C-1)/2$，占比 $(C-1)/(2C)$：一半的块在 $C$ 大时消失，而 `comm bytes` 一分
不省——这正是"省算力不省带宽"的量化版本。注意对照组必须处理"块里一个 key 都没剩下"的情况
（`max()` 空序列会抛 `ValueError`），也就是说**没有整块跳过，就得逐行判空**；本题的跳过是把
逐行判空提前到块级。

在线 softmax 的必要性同样能量出来：不扣最大值的朴素写法在 logit 稍大时自己就崩了。

```text
# 4 ranks, shard length 8, head width 4, entries in [-1, 1], 20 random trials
# blown = rows where the naive exp(logit) form overflows or divides 0 by 0
  scale   max|logit|   naive rows blown up   max|online - naive| (surviving)
    1.0          3.3           0/640         2.50e-16
    5.0         16.5           0/640         7.29e-16
   20.0         66.1           0/640         2.86e-15
   50.0        165.4           0/640         3.66e-15
  200.0        661.4           0/640         1.13e-14
  500.0       1653.6         134/640         2.43e-14
 2000.0       6614.5         596/640         1.26e-04
```

`scale=500` 起朴素写法开始 `OverflowError`（`exp(1653)` 已经超出双精度），`scale=2000` 时 640 行里
有 596 行算不出来；侥幸没崩的那些行也被相消吃掉了 4 个有效数字（$1.26\times10^{-4}$）。在线写法
全程把指数压在 $\le 0$，误差始终在 $10^{-14}$ 量级，并且 $l \ge 1$ 保证了不会除零。

## 范围与来源

- **不做归属声明**。Ring Attention 是 Liu 等（2023）的算法，`ring-flash-attention` 等开源实现
  确实把 KV 沿环传递，但离线环境里没有任何一份可对照的实现（`.venv-eval` 只有 torch / numpy /
  transformers）。本题的 `src = (r - s) mod C`、"整块跳过判据"、"`comm` 只在 $C-1$ 步计"、
  "`traces` 被跳过的步照抄"、"`scale` 由调用方给定而不是 $1/\sqrt d$" 全部是**题面自定**的
  一套自洽口径。真实系统在这些点上都不一样：KV 按 head 或按层分块转、用 zigzag 曲折编号做
  负载均衡、块大小可以与 rank 数无关、通信与计算 overlap。
- **性质靠自测背书**：跳过占比 $(C-1)/(2C)$ 是组合恒等式（被吃掉的块数 $C(C-1)/2$ ÷ 总块数
  $C^2$），脚本对 $C=2,4,8,16$ 逐一验证了这个式子，同时验证了"跳过与不跳过输出**逐位相同**"。
- **与相邻题的分工**：`flash-attention-tile` 是单卡分块在线 softmax，没有环、没有通信账、
  也没有整块跳过；`distributed-sequence-parallel` 只算闭式的显存与通信字节，不模拟每一步；
  `distributed-ring-allreduce-steps` 是 Ring All-Reduce 的步数快照，机制与注意力无关；
  `distributed-all-to-all` / `distributed-tp-split-rearrange` 走的是 Ulysses 式的全交换路线，
  不是环上轮转。本题填的是"环转 + 在线递推 + 因果整块跳过"这一格。
