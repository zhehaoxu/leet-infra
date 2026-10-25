# 梯度压缩带宽账：1-bit 量化与低秩投影

## 题目描述

数据并行每一步都要做一次 all-reduce，**通信量正比于梯度字节数**，跨节点时网卡（几十 GB/s）往往比 NVLink
慢一个数量级，于是「少发点字节」成了最直接的优化：**1-bit Adam / 符号量化（sign compression）** 把每个元素压到
1 位（再给一组元素配一个 fp32 scale），**PowerSGD** 则把每个矩阵写成两个低秩因子 $P Q^{\top}$，只发这两块。
两者都要靠 **error feedback**（本地 fp32 残差累加器）把丢掉的量攒到下一步，因此**省带宽但不省显存**。
压缩也不是稳赚：位数一高、分组一细、秩一接近满秩，压缩后的载荷就会**反超未压缩的 fp32**，工程上必须
切回原始 all-reduce（**回退**）。本题把这套决策写成一张纯字节账。

`distributed-zero-communication` 算的是 ZeRO 在**未压缩**梯度下的通信量，`operator-fusion` 只算 kernel 的
HBM 读写字节，`transformer-mem` 只算参数量对应的字节；本题只管**压缩后跨卡搬运的字节、省下的比例，
以及回退/最大可行位数这两个切换点**。

## 规则

- **载荷**（整数向上取整，$numels=0$ 的矩阵整块跳过）：`full` 按总元素乘每元素字节；`sign` 每矩阵是
  「符号位按位打包 $\lceil numels\cdot bits/8\rceil$」加上「分组 scale $\lceil numels/g\rceil\cdot 4$」，
  $g=0$ 时整块张量只有一个 scale（记 4 字节）；`lowrank` 每矩阵发 $(rows+cols)\cdot r$ 个元素，
  但 $r\ge\min(rows,cols)$ 时**按满秩处理**，直接发原矩阵。
- **回退**：`compressed = min(payload, full_payload)`，`fallback = payload > full_payload`（持平不回退）。
- **环系数**沿用 `distributed-zero-communication`：$c=(W-1)/W$，一次 all-reduce 每 rank 搬 $2cS$，
  因此 $W=1$ 时两个通信量都是 $0$。`saved_ratio` 按载荷之比算，与 $W$ 无关。
- **`max_bits`**：只有 `method="sign"` 需要求，把位数从 1 扫到 `bytes_per_element*8`，取仍然
  满足 `payload <= full_payload` 的**最大**位数（载荷对位数单调不减，扫不到就记 0，
  `full_payload == 0` 也记 0）；`full` / `lowrank` 恒为 0。
- 具体的输入含义、每条公式与返回元组的类型口径全部写在 `starter.py` 的 docstring 里，以它为准。

## 示例

```
输入：shapes=[[4096,4096]], world=8, method="sign", bytes_per_element=4, bits_per_element=1
  full_payload = 4096*4096*4 = 67,108,864
  payload = ceil(16,777,216*1/8) + 4 = 2,097,152 + 4 = 2,097,156（g=0：整块只有一个 scale）
  c = 7/8 -> per_gpu = 2c*2,097,156 = 3,670,023.0    full_comm = 2c*67,108,864 = 117,440,512.0
  saved = 1 - 2,097,156/67,108,864 = 0.9687499403953552   ef = 16,777,216*4 = 67,108,864
  max_bits = 31（第 32 位时 2,097,152*32 + 4 已经大于 67,108,864）
输出：(67108864, 2097156, 3670023.0, 117440512.0, 0.9687499403953552, 67108864, False, 31)

输入：shapes=[[1280,3584]], world=8, method="lowrank", bytes_per_element=2, rank=8
  full_payload = 1280*3584*2 = 9,175,040   payload = (1280+3584)*8*2 = 77,824
  per_gpu = 1.75*77,824 = 136,192.0        full_comm = 1.75*9,175,040 = 16,056,320.0
  saved = 1 - 77,824/9,175,040 = 0.9915178571428571   ef = 4,587,520*4 = 18,350,080
输出：(9175040, 77824, 136192.0, 16056320.0, 0.9915178571428571, 18350080, False, 0)

输入：shapes=[[1024,1024]], world=4, method="sign", bytes_per_element=1, bits_per_element=8,
      scale_group_size=128
  full_payload = 1,048,576（fp8 基线）   payload = 1,048,576 + ceil(1,048,576/128)*4 = 1,081,344
  payload > full -> fallback=True, compressed=1,048,576, saved=0.0, 两个 comm 量都等于 1,572,864.0
  max_bits = 7（第 8 位就追平基线，再加 scale 必然超）
输出：(1048576, 1048576, 1572864.0, 1572864.0, 0.0, 4194304, True, 7)
```

## 约束

- `shapes` 长度 $\ge 1$，每项是 $[rows, cols]$（$\ge 0$，允许出现 $0$ 表示该块不参与通信）；
  `world` $\ge 1$；`method` $\in$ `{"full","sign","lowrank"}`；`bytes_per_element` $\ge 1$；
  `bits_per_element` $\ge 1$；`scale_group_size` $\ge 0$；`rank` $\ge 1$
- 返回八元组 `(full_payload, compressed_payload, per_gpu_comm_bytes, full_comm_bytes, saved_ratio,
  ef_state_bytes, fallback, max_bits)`：第 1、2、6、8 项是 int（`torch.equal` 口径的精确整数），
  第 3、4、5 项是 float（`atol=1e-5, rtol=1e-5`），第 7 项是 bool
- 纯字节账：不建模压缩带来的收敛精度损失，也不模拟 top-k 稀疏化或梯度分片归属
