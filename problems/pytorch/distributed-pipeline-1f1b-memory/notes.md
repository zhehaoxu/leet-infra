# 解题思路

## 面试官意图

「为什么上了 1F1B？」「为什么流水线越深、micro-batch 越多，激活显存反而不爆？」——答案是 1F1B 把每个 stage
同时在飞的 micro-batch 数从 GPipe 的 $m$ 压到 $\min(m, p-i)$。能写出这个峰值公式，就说明你懂 PP 的**显存**
那一半（`distributed-pipeline-bubble` 管的是**时间**那一半）。

## 每段在飞数怎么来的

1F1B 稳态是「一次前向紧跟一次反向」。warmup 阶段，越靠输入端的 stage 要先跑越多前向等着反向，第 `i` 段
最多囤 $\min(m,\,p-i)$ 个 micro-batch 的激活：
- 输入端 $i=0$：$\min(m,p)$（最重）。
- 输出端 $i=p-1$：$\min(m,1)=1$（最轻，来一个做一个反向立刻释放）。

GPipe 没有这种交错：所有 $m$ 个前向先跑完，每段都囤满 $m$ 份。

## 省了多少（考点）

$$ \frac{\text{offfb\_worst}}{\text{gpipe}} = \frac{\min(m,p)}{m} = \begin{cases} 1 & m\le p\\ p/m & m>p \end{cases} $$

- $m \le p$：**最重段一点没省**（还在 warmup 就把所有 m 前向跑完了）；只有输出端段省到 1 份。
- $m > p$：最重段也降到 $p$ 份，比值 $p/m$。所以「加深流水线 $p$ 反而涨显存」——$p$ 既受气泡（要 $m\gg p$）
  又受显存（要 $p$ 小）双向夹逼，这正是 PP 度数难定的根因。

## 实现要点

$lps = L//p$（要求整除）、$\text{per\_mb}=\text{lps}\cdot\text{act\_bytes}$，再乘各档在飞数。判题独立参考用
**逐段 profile 取 max/min** 得到最重/最轻段（等价于闭式 $\min(m,p)$/$\min(m,1)$，两条路径互验）。

## 相关题

`distributed-pipeline-bubble`（PP 气泡/时间）、`training-state-memory`（参数/梯度/优化器显存，与激活相加才是
总显存）。激活本身可再用 `transformer-mem` 的梯度检查点进一步压缩——那是与调度正交的另一维。
