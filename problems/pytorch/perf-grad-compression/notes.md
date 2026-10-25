# 详解：梯度压缩的字节账、回退切换点与最大可行位数

## 出处

- **1-bit Adam / 符号量化**：Li 等, *1-bit Adam: Communication Efficient Large-Scale Training of Adam's At Scale*
  （ICML 2021，arXiv:2102.02895；微软 DeepSpeed 的 `1bit-adam` 实现）。工程实现里符号位之外还要给一组元素（如
  148×1024 个）配一个 fp32 的 scale，本题用 `scale_group_size` 把这个分组粒度做成显式参数。
- **PowerSGD**：Vogels, Karimireddy, Jaggi, *PowerSGD*（ICML 2019）：把每个梯度矩阵写成秩 $r$ 的两个因子
  $P\in\mathbb{R}^{rows\times r}$、$Q\in\mathbb{R}^{cols\times r}$，通信量从 $rows\cdot cols$ 降到
  $(rows+cols)\cdot r$。
- **error feedback**：Karimi 等, *Error Feedback Fixes Signum-gradient and Other Gradient Compression Schemes*
  （MLSys 2019）：压缩丢掉的残差要在本地 fp32 累加，因此**省的是带宽、不是显存**。
- 环 all-reduce 的通信系数 $c=(W-1)/W$、每 rank 搬 $2cS$ 沿用题库 `distributed-zero-communication` 的口径。

## 核心与易错点

1. **位打包要向上取整**：$\lceil numels\cdot bits/8\rceil$ 是整数进位，不是浮点除法。`bits=3`、
   $numels=100$ 时是 $\lceil 37.5\rceil=38$ 字节；用 `int(...)` 截断会在所有非 8 倍数的用例上少算。
2. **`scale_group_size=0` 是「整块张量共用一个 scale」，不是「没有 scale」**：每组 4 字节、**每块矩阵各付一次**。
   $g=1$ 时 scale 开销就是 $numels\cdot 4$，已经等于 fp32 全量，再加符号位必然回退；$g=numels$ 与 $g=0$ 完全等价，
   这是判题钉口径的退化点。
3. **低秩是「线性」开销，且有满秩上限**：载荷 $(rows+cols)\cdot r\cdot bpe$（两块因子相加，不是 $r^2$ 也不是
   $r\cdot\min(rows,cols)$）。$r\ge\min(rows,cols)$ 时按满秩处理**直接发原矩阵**；$r=\min-1$ 时
   $(rows+cols)(r)$ 常常**大于** $rows\cdot cols$，比如 $1024\times1024$、$r=1023$ 得到 8,379,904 字节 > fp32 的
   4,194,304 —— 这就是「压缩反而更贵」，要回退。
4. **回退判据用严格大于**：`payload == full_payload` 时 `compressed == full`、`fallback=False`、`saved_ratio=0.0`；
   写成 `>=` 会让 rank 恰好满秩、或 `method="full"` 的用例多报一个 True。
5. **环系数 $c=(W-1)/W$**：`per_gpu_comm_bytes = 2cS`，$W=1$ 时是 $0$（单机不需要 all-reduce），$W\to\infty$ 才趋近
   $2S$。$W$ 翻倍不等于搬运量翻倍（$1.5\to1.75$）。`saved_ratio` 是载荷之比，与 $W$ 无关，别把 $2c$ 也乘进去。
6. **error feedback 恒按 fp32**：`ef_state_bytes = total_elements*4`，与 `bytes_per_element`、`method`、位数都无关；
   只有 `error_feedback=False` 才是 0。1-bit Adam 的显存瓶颈从来不在通信侧。
7. **`max_bits` 是「仍不回退」的最大位数**：载荷对位数单调不减，所以正向扫到底取最后一个可行值即可，
   不需要二分（元素数很小时二分反而写错边界）。它经常等于 `bytes_per_element*8 - 1`，因为那 4 字节 scale
   刚好吃满最后的余量；非 sign 方法恒记 0，`full_payload == 0`（空形状清单或全 $0$ 尺寸）也记 0。
8. **空矩阵整块跳过**：$numels=0$ 时既不打包也不付 scale，写成「每块至少 4 字节」会让
   `shapes=[[0,64],[64,0]]` 多算出 8 字节。
9. **分工**：未压缩通信量与分片归属在 `distributed-zero-communication` / `distributed-zero-shard-ownership`，
   参数与优化器状态的**字节**在 `training-state-memory` / `transformer-mem`，HBM 流量在 `operator-fusion`，
   本题只管压缩之后跨卡搬运的**字节**与切换点。

## 对拍口径

参考解：`_sign_payload` 用 `(a+b-1)//b` 做向上取整、低秩写成 `(rows+cols)*rank*bpe`、环系数写成
`2.0*(W-1)/W`、`max_bits` **从 1 正向扫**。判题 `_oracle` 全部换路：full 载荷按 8 行分块累加
（`min(8, rows-start)*cols*bpe`），位打包用 `divmod(n*bits, 8)` 的余数判进位，分组数用 `(n-1)//g+1`，
低秩拆成 `rows*rank*bpe + cols*rank*bpe` 两个乘积，环系数写成 `payload*(2 - 2/W)`，`saved_ratio` 用整数差
`(full-compressed)/full`，`max_bits` **从最高位向下扫**取第一个可行值（方向相反、结论必须一致）。
另设结构性断言：$W=1$ 时两个通信量为 0、$W\in\{2,4,8\}$ 时系数为 $1/1.5/1.75$、`g=1` 时必回退且
`max_bits=0`、$r=\min$ 恰好不回退而 $r=\min-1$ 必回退、`compressed <= full` 且 `per_gpu <= full_comm` 的
72 组网格不变量、空形状清单全 0、`error_feedback=False` 只改第 6 项。

## 复杂度

参考解 $O(K + K\cdot B)$，$K$ 为矩阵块数、$B=$ `bytes_per_element*8`（`max_bits` 的线性扫描）。
判题的 oracle 分块循环是 $O(\sum rows/8)$，整套用例几十毫秒内完成。
