# 集合通信：Reduce-Scatter

## 题目描述

**Reduce-Scatter** = 先对所有 rank 的数据做逐元素**归约**（本题为求和），再把归约结果**切片分发**：
rank `i` 只拿第 `i` 段。它是 Ring All-Reduce 的前半程，也是 ZeRO 里「每卡只保留 $1/W$ 梯度分片」的
通信原语。

> 集合通信按纯函数建模：输入是「通信前各 rank 的本地数据」列表（第 $r$ 项是 rank $r$ 的张量），
> 输出是「通信后各 rank 的数据」列表（第 $r$ 项是 rank $r$ 拿到的那一段）。

## 规则

- 实现 `solve(inputs)`，`inputs` 是长度 $W$ 的列表，`inputs[r]` 是 rank `r` 的一维张量，各 rank **等长**
  且长度能被 $W$ 整除。
- 两步：
  1. **Reduce**：逐元素求和 $T = \sum_{r=0}^{W-1} \texttt{inputs[}r\texttt{]}$（长度 $L$）。
  2. **Scatter**：把 $T$ 均分成 $W$ 段（每段 `chunk = L / W`），rank `i` 得到第 `i` 段
     $T[i\cdot\text{chunk} : (i+1)\cdot\text{chunk}]$。
- 返回**长度 $W$ 的列表**，第 `i` 项是 rank `i` 分到的那段（**各 rank 结果不同**，这正是与 All-Gather 的区别）。
- 不得修改输入张量。

## 示例

```
输入：inputs = [ [1,2,3], [4,5,6], [7,8,9] ]   # W=3, 每个长度 3, chunk=1
  Reduce  T = [1+4+7, 2+5+8, 3+6+9] = [12, 15, 18]
  Scatter rank0←T[0:1]=[12], rank1←T[1:2]=[15], rank2←T[2:3]=[18]
输出：[ [12], [15], [18] ]
```

## 约束

- `1 ≤ W ≤ 64`；各 rank 一维、等长、长度能被 $W$ 整除
- 返回 `list[torch.Tensor]`，长度 $=W$，每项形状 $=(L/W,)$
- 用 `torch.equal` 逐段判等；注意 rank 顺序：第 `i` 项对应 $T$ 的第 `i` 段
