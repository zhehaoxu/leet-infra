# 解题思路

## 面试官意图

很多人误以为 ZeRO「既省显存又省通信」。恰恰相反：ZeRO-1/2 的通信量和 DDP **完全一样**，ZeRO-3 反而
**更多**。能讲清「分区带来了什么额外搬运」是区分「用过 ZeRO」和「懂 ZeRO」的关键。本题用一层简化模型
把这个 tradeoff 量化。

## 环系数 c 的来历

Ring 系列原语每 GPU 收发都带 $c=\frac{W-1}{W}$：做一次 All-Reduce（= Reduce-Scatter + All-Gather）每 GPU
搬运 $2cS$；只做其中一个方向是 $cS$。当 $W$ 很大时 $c\to 1$，于是「All-Reduce ≈ 每卡搬两份数据」是好记的
近似。

## 三个方案逐项拆

| 方案 | 通信构成 | 每 GPU 字节 |
|---|---|---|
| DDP | 梯度 All-Reduce | $2cG$ |
| ZeRO-1/2 | 梯度 Reduce-Scatter + 更新后参数 All-Gather | $cG + cG = 2cG$ |
| ZeRO-3 | 上一行 + 前向/反向各 All-Gather 一份参数 | $2cG + 2cP$ |

- ZeRO-1/2 只是把「优化器状态/梯度放在哪」改了，梯度同步 + 参数回传两步合起来仍等于一次 All-Reduce，
  故与 DDP 持平。
- ZeRO-3 因为参数本身也切片存放，前向要临时把本层参数收拢、反向再收拢一次，多搬 $2cP$。

## 实现要点

先算 $c$（注意 $W=1$ 时 $c=0$ → 全零）、$G=N\cdot\texttt{grad\_bytes}$、$P=N\cdot\texttt{param\_bytes}$，
再按上表组合。`check_zero_equals_ddp`、`check_zero3_heavier` 就是考这两个反直觉结论有没有被真正写对。

## 与显存题的关系

`training-state-memory` 给出 ZeRO-1/2/3 的**每卡显存**（$16N \to 16N/W$）；本题给出对应的**每卡通信**。
两题合起来才是完整的 ZeRO tradeoff：**用通信换显存**，stage 越高显存越省、通信越贵。

## 判题口径

`test_cases.py` 用分步累加的独立参考 `_ref` 对拍，并校验 ZeRO-1/2≡DDP、ZeRO-3 严格更贵、$W=1$ 无通信、
dtype 字节翻倍线性。
