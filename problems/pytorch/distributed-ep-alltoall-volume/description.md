# 分布式：MoE EP all-to-all 通信量与 straggler 账

## 题目描述

专家并行（Expert Parallelism，EP）中，rank 根据路由表把 token 交给持有目标专家的 rank。`routing[r, e]` 表示源 rank \(r\) 路由给专家 \(e\) 的 token 数；专家固定放在 rank \(e\bmod W\)，专家数不必被 rank 数整除。

为**单次 dispatch all-to-all**建立通信账本，计算路由计数、各 rank 的网络收发字节、不均衡度和带宽模型下的耗时。所有路由计数都按给定值执行，不建模 token 丢弃。

## 通信与耗时规则

发送计数矩阵为

\[
S_{rq}=\sum_{e:\ e\bmod W=q}\text{routing}[r,e],
\qquad R=S^\mathsf{T}.
\]

\(S\) 和接收矩阵 \(R\) **保留对角项**，但对角项是本地专家路由，不经过网络。设每个 token 为 \(b\) 字节，则 rank \(r\) 的网络发送加接收总字节为

\[
B_r=\left(\sum_q S_{rq}+\sum_q S_{qr}-2S_{rr}\right)b.
\]

本地项在行和与列和中各出现一次，所以要减去两次。跨 rank 传输在发送端和接收端各计入一份负载。

令 \(M=\max_r B_r\)，\(\mu=\frac{1}{W}\sum_r B_r\)。不均衡度为 \(M/\mu-1\)；全部网络负载为零时取 0。

`bandwidth` 是每个 rank 的**网络发送加接收总带宽，单位字节/秒**，因此估算耗时为 \(M/\text{bandwidth}\)。这不是以 `max(发送, 接收)` 计算的全双工模型，也不是 dispatch 加反向 combine 的总耗时。忽略计算延迟、通信启动延迟、元数据和反向 combine。完整类型与合法输入约定见 starter。

## 示例

```text
routing=[[1, 2, 3], [4, 5, 6]]
bytes_per_token=4, bandwidth=1e9
send_counts=[[4, 2], [10, 5]]
recv_counts=[[4, 10], [2, 5]]
per_rank_bytes=[48, 48]
max_load_bytes=48, imbalance=0.0, wall_clock=4.8e-8 秒
```

专家 0、2 属于 rank 0，专家 1 属于 rank 1。跨 rank token 数分别为 2 和 10，因此两个 rank 各处理 \((2+10)\times4=48\) 网络字节；本地的 4 和 5 个 token 不计网络负载。输出中前三项张量以列表展示：

```text
([[4, 2], [10, 5]], [[4, 10], [2, 5]], [48, 48], 48, 0.0, 4.8e-8)
```

单 rank、全本地路由或全零路由的网络字节和耗时均为零，即使发送矩阵对角线上存在非零计数。
