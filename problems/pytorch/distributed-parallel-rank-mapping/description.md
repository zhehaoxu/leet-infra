# 分布式：并行网格 rank 映射与通信组

## 场景

总共 $W$ 个 rank 按 TP、CP、DP、PP 等独立轴组成网格。同一轴通信组内只改变该轴坐标，其它坐标保持不变。需要输出各 rank 的坐标、各轴通信组成员、组内 local rank，以及每个轴独立编号的组号。

## 低位轴优先编码

轴顺序由 `order` 指定，**越靠前变化越快**，而不是由字典的插入顺序决定。记第 $j$ 个轴的大小为 $d_j$，步长为 $s_j=\prod_{k<j}d_k$：

$$rank=\sum_j c_j s_j,\qquad c_j=\lfloor rank/s_j\rfloor\bmod d_j.$$

例如 `['tp','cp','dp','pp']` 表示

$$rank=c_{tp}+tp\,c_{cp}+tp\,cp\,c_{dp}+tp\,cp\,dp\,c_{pp}.$$

这是**最低有效轴在前**的混合进制编码。常规张量行主序是末轴最快，不能直接用相同顺序的 shape 去 reshape；若借助行主序网格，应先反转轴顺序。

## 组归属

对每个轴：

1. 固定所有其它轴，枚举该轴坐标，得到成员并按 rank 升序排列。
2. local rank 是当前 rank 在成员列表中的下标，等于该轴坐标。
3. 将该轴的不同通信组按最小成员升序排列，依次编号为 $0,1,\ldots,W/d_j-1$。组号只在该轴内有意义，不跨轴共享。

网格必须完整覆盖全部 rank，轴顺序必须恰好包含每个轴一次。

## 示例

```text
world_size=8, dims={'tp':2,'dp':4}, order=['tp','dp']
rank 0: coord={'tp':0,'dp':0}
        groups={'tp':[0,1], 'dp':[0,2,4,6]}
        local_ranks={'tp':0,'dp':0}, group_ids={'tp':0,'dp':0}
rank 1: coord={'tp':1,'dp':0}
        groups={'tp':[0,1], 'dp':[1,3,5,7]}
        local_ranks={'tp':1,'dp':0}, group_ids={'tp':0,'dp':1}
```

这是纯整数模拟，不创建真实通信进程组，也不处理 EP 与 DP 共享 rank 维度等非独立网格布局。
