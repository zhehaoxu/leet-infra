# 解题思路

## 面试官意图

「数据并行加卡为什么会遇到通信瓶颈？为什么用 Ring All-Reduce 而不是中央参数服务器？」这类问题
的标准答案就是本题的 $2\cdot\frac{W-1}{W}S$。它是分布式训练通信分析的基石，也常和 `training-state-memory`、
ZeRO 一起被问。

## 为什么是 $2\cdot\frac{W-1}{W}S$

Ring 把 $W$ 张卡连成一个环，All-Reduce 拆成两半做：

1. **Reduce-Scatter**：$W-1$ 步，每步每卡把 $\frac1W$ 的数据发给下一个并累加，结束时每卡手里有
   $\frac1W$ 的**完整求和结果**。发送量 $=(W-1)\cdot\frac{S}{W}$。
2. **All-Gather**：再 $W-1$ 步把这些局部结果转发一圈，让每卡拿到**全部** $\frac WW$ 段。同样的量。

两段相加：$2\cdot\frac{W-1}{W}\cdot S$。

## 关键洞察：与卡数近乎无关

系数 $\frac{W-1}{W}$ 随 $W$ 增大快速趋近 $1$，所以每卡传输量封顶在 $2S$，**不随卡数线性膨胀**。
对比 naive 方案：参数服务器要让中心节点收发 $2S\cdot W$，卡越多 server 越堵。这就是 Ring 能 scale
到成百上千卡的根本原因（带宽项主导时）。

## 延迟项的作用

$2(W-1)$ 个跳各带一个固定 start-up 延迟，小消息 / 超大集群时延迟项 $\propto W$ 会反超带宽项成为瓶颈——
这也是实践中用 **double binary tree / halving-doubling** 替代纯 ring 的原因。本题用 `latency_per_hop`
参数让你把这项也算进去（默认 0 只看带宽）。

## 常见错误

- 漏掉系数 2（只算了 reduce-scatter 或只算 all-gather）。
- 把每卡传输量写成 $2S\cdot(W-1)$（少了除以 $W$），那就退化成 server 的线性膨胀了。
- $W=1$ 没特判，返回了非零通信量。

## 判题口径

`test_cases.py` 用**分两阶段累加**的独立参考核对闭式，并固化三个不变量：$W=1$ 零通信、每卡传输量
$\le 2S$ 且随 $W$ 单调趋近 $2S$、延迟项只加在时间上而不改体积。
