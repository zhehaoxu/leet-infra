# 详解：kernel 发射时间线与 CUDA Graph

## 出处

PyTorch 官方 **CUDA semantics / CUDA Graphs** 文档（`torch.cuda.CUDAGraph`、`torch.compile(mode="reduce-overhead")`）
明确说明：每次 launch 要在 CPU 上付若干微秒的固定开销，短 kernel 密集时 GPU 会「等命令」而空转，把一串 kernel
打包成 graph 后 replay 只付一次开销。本题把这段描述变成一条可算的时间线（单机单流的简化模型）。

## 核心与易错点

1. **就绪时刻是 $(i+1)L$ 不是 $iL$**：CPU 从 $t=0$ 开始发射，第 0 个 kernel 也要等自己这一次发射完成才能起跑。
   写成 $iL$ 会让首 kernel 凭空提前 $L$，$\text{wall}$ 与空转个数都错。
2. **单流串行递推**：$s_i=\max(r_i,e_{i-1})$，$e_i=s_i+d_i$。GPU 不能并行跑两个 kernel，所以后一个只能等前一个
   结束；「CPU 早发完了」并不额外帮忙，只是不阻塞。
3. **busy 的分母是墙钟不是有效时长**：$\text{busy}=W/\text{wall}$，恒 $\le1$；$\text{wall}=W+\text{空转}$，所以
   `busy_ratio * wall == sum(durations)` 是本题的恒等式探针。
4. **launch-bound 计数按 eager 口径**：它是「eager 时间线里有多少 kernel 被发射卡住」，与 `use_graph` 无关——
   打包之后并没有 kernel 在等发射，但作为诊断量仍按 eager 报告。$L>0$ 时它至少为 1（首发射必然造成一次空转）。
5. **graph 收益的上界**：graph 墙钟 $=L+W$，于是 $\text{speedup}=\text{eager}/(L+W)$。kernel 全都远长于 $L$ 时
   $\text{eager}=L+W$，speedup 恰为 1（图救不了 compute/带宽瓶颈）；$d_i\equiv 0$ 时 speedup $=n$（纯发射账）。   这两个退化点正是判题里的两条硬断言。
6. **别把 kernel 时长当 launch 时长**：$d_i$ 是 GPU 执行、$L$ 是 CPU 发射，两者量纲相同但来源不同。

## 对拍口径

参考解正向递推 $s_i/e_i$；判题 `_oracle` 用「单机 + 释放时间」的经典 max-plus 闭式
$e_i=\max_{j\le i}\big[(j+1)L+\sum_{k=j}^{i}d_k\big]$，对每个起点暴力枚举（三重循环），结构与递推完全不同；
空转个数再由 $e_{i-1}$ 与 $r_i$ 比较得出。另设 6 组退化断言：$n=1$、$L=0$、全零时长、全零 + 零开销（两处除零兜底）、
长 kernel（只有首发射空转、speedup=1）、`busy*wall == sum(d)` 恒等式，以及 graph 恒不慢于 eager。

## 复杂度

参考解对 $n$ 个 kernel 一次遍历：$O(n)$ 时间、$O(1)$ 额外空间（列表转 float 为 $O(n)$）。
判题暴力 oracle 为 $O(n^3)$，用例 $n\le 6$，秒内完成。
