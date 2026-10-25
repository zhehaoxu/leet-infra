# 详解：序列/上下文并行显存-通信账

## 出处
Korthikanti et al. 2022（arXiv:2205.05198）的 **sequence parallelism**：把 LayerNorm/Dropout 等区域的激活沿**序列维**切开、与张量并行配合，切分区与非切分区之间用 All-Gather / Reduce-Scatter 转换。超长上下文的 **Context Parallel**（Ring Attention，Liu et al. 2023, arXiv:2304.03278）同样沿序列维分片、让 KV 在各 rank 间环形流转。本题用一个自定的简化模型量化「切分省多少显存 / 多花多少通信」，环系数口径与 `distributed-zero-communication` 一致。

## 核心与易错点
1. **两部分激活**：只有「可切分」的 $f\cdot A$ 会被 $C$ 个 rank 分摊；剩下 $(1-f)\cdot A$ 每卡都要整份复制。所以每卡显存**不是** $A/C$，而是 $(1-f)A + fA/C$——$f<1$ 时有下限 $(1-f)A$，加再多卡也切不动那部分。
2. **通信是两次原语**：进入全序列视角做一次 All-Gather、回去做一次 Reduce-Scatter，故系数是 $2$，不是一次。每 rank 搬 $2 f A\cdot\frac{C-1}{C}$。
3. **环系数 $\frac{C-1}{C}$**：$C$ 越大越接近 1，「每卡约搬两份切分数据」是好记近似；$C=1$ 时系数为 0，退化单卡零通信。别把系数写成 $\frac{C}{C-1}$ 或漏掉 $C=1$ 除零。
4. **$f=0$ / $f=1$ 边界**：$f=0$ → 全复制、无通信；$f=1$ → 每卡 $A/C$、通信 $2A\frac{C-1}{C}$。$A=0$ 恒为 0。
5. **单调权衡**：$C\uparrow$ → 每卡显存 $\downarrow$、通信 $\uparrow$，正是「用带宽换显存」。

## 对拍口径
参考解走闭合式 $(1-f)A+fA/C$ 与 $2fA(C-1)/C$；判题 `_oracle` 用**环上逐步模拟**——把可切分量 $S=fA$ 拆成每步 $S/C$，reduce-scatter 与 all-gather 各累加 $C-1$ 步得 $2(C-1)S/C$，结构不同、数值等价（浮点用相对容差 `_close`）。另设 $C=1$ 退化、$f\in\{0,1\}$ 边界、随 $C$ 单调三组断言。

## 复杂度
\(O(1)\) 闭合式；oracle 环模拟 \(O(C)\)。
