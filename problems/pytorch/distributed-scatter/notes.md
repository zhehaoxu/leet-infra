# 解题思路

## 面试官意图

Scatter / All-Gather 是一对互逆操作，常被拿来一起考。「把一批数据切分发到各卡」和「把各卡的分片
收拢回来」如果分不清，就会在数据并行加载、张量并行布局时写反方向。本题是 All-Gather 的镜像，
实现越简单，越要留意语义细节。

## 语义

源 rank 持有一维完整数据 `full`（长 $L$，能被 $W$ 整除）。Scatter 把它**均分 $W$ 段**，第 `i` 段
`full[i*chunk:(i+1)*chunk]` 只发给 rank `i`（`chunk = L/W`）。输出是**每 rank 一段、各不相同**的列表。

## 与相邻原语的对照

- **All-Gather**：各 rank 一片 → 人人拿到拼接的全体（Scatter 的逆）。
- **Scatter**：一份完整 → 各 rank 一片（**不求和**）。
- **Reduce-Scatter**：各 rank 一份完整 → 先求和再各拿一段。
  三者最容易混的就是 Scatter vs Reduce-Scatter：前者输入只有一份数据、不做跨 rank 归约。

## 实现要点

`full.reshape(-1)` 后按 `chunk` 步进切片即可；用 `.clone()` 让每段独立、且不共享 `full` 存储以满足
「不修改输入」。测试 `check_inverse_of_allgather` 固化了 **`cat(scatter(full)) == full`** 这条互逆不变量。

## 真实场景

数据并行里 `DistributedSampler` 把整个数据集按 rank 切分下发，本质就是一次 scatter 的思想：每个 worker
只取自己那 $\frac1W$ 的样本。张量并行初始化时也常用 scatter 把某个大权重按维切成每卡一份。

## 判题口径

`test_cases.py` 校验：切片顺序正确、`cat` 拼回等于原数据、`chunk=1` 与 `world=1` 退化、
多种 $(L, W)$ 组合逐段对拍、不修改输入。
