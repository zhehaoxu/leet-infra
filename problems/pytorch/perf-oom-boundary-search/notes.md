# 详解：把显存预算反解成边界——floor、二次正根与整除判定

## 出处

- **峰值显存的组成**沿用题库口径：`training-state-memory` / `transformer-mem` 给出「参数 2 + 梯度 2 +
  master/m/v 12 = 16 字节/参数」的静态占用，激活的 $O(s)$ 与 $O(s^2)$ 两项来自标准 Transformer 反向需要保留的
  中间量（MLP/残差是线性项，注意力分数矩阵是 $s^2$ 项）——这也是 FlashAttention 之所以能改命的原因：
  把 $s^2$ 项系数打到 0。
- **按 micro batch / 序列长反解**是 Megatron-LM、DeepSpeed 等框架调参的常规动作（`micro-batch-size`、
  `max-seq-length`、`global-batch-size = micro * accum * world_size`），本题把这条等式做成整数判据。
- **二次不等式求最大整数根**是经典的二分/牛顿场景，这里刻意保留「闭式 + 用判据校正 ±1」的写法，
  和判题的纯整数二分互为验证。

## 核心与易错点

1. **两种「占用的钱」不一样**：`reserve_bytes` 从**额度**里扣（usable = hbm − reserve，不进峰值），
   `cuda_overhead_bytes` 进**峰值**（每次求 peak 都要加上）。把两者混在一起，两条边界会同时错一格。
2. **可行性是闭区间**：`peak <= usable`，**刚好用满算可行**；判题专门造了 `peak == usable` 的用例，
   写成 `<` 就会把「刚好放得下」判成 OOM。`headroom` 允许为负且**不 clamp**（负值本身就是「还差多少」的答案）。
3. **二次项也乘 m**：$\text{peak}$ 的最后一项是 $m\cdot s^2\cdot apq$，不是 $s^2\cdot apq$——每条序列都有
   自己的一份注意力分数。少了 $m$，`max_micro_batch` 会被系统性高估。
4. **`max_micro_batch` 是线性 floor**：预算除以「每 micro batch 的成本」$s\cdot ap+s^2\cdot apq$，
   整数除法天然正确，不用搜索；而 **`max_seq` 是二次的**：$m\,apq\,s^2+m\,ap\,s\le\text{budget}$，
   正根 $\frac{-b+\sqrt{b^2+4aR}}{2a}$ 取 floor 后**必须代回判据校正**（浮点开方可能差一个 ulp），
   否则会给出 `bound+1` 这种「代回去就超」的错答案。`apq == 0` 时它退化成线性 floor，别开平方。
5. **先判预算正负**：$\text{budget}=\text{usable}-\text{cuda\_overhead}-\text{static}<0$ 时连固定开销都放不下，
   两条边界都是 $0$（题目约定「连 1 都放不下记 0」），此时 `world_size_needed` 仍然照常算，`global_batch_fits`
   因为 feasible=False 而必为 False。
6. **`world_size_needed` 用给定的 m，不是 `max_micro_batch`**：目标是「当前这套配置需要几个 rank」。
   例：$A=8, m=4$ 时每个 rank 一步吃 32 条，$G=120$ 虽然能被 $8\times 15$（15 恰是 `max_micro_batch`）整除，
   但 $120/32$ 不整除 -> 需要 4 个 rank、而 $4\times32=128\ne 120$ -> `global_batch_fits=False`。
7. **ceil 之后还要回乘**：$\text{world}=\lceil G/(A m)\rceil$ 只保证「够装」，判定凑得齐要用
   $\text{world}\cdot A\cdot m == G$（或等价的 $G\bmod(A m)==0$）。只做整除判据而不取 ceil，会在
   $G<A m$（如 $G=31, A m=32$）时误判。
8. **`act_per_token >= 1` 是有界性的保证**：线性项保证两条边界一定有上界，因此不需要「无界记 -1」这类
   额外哨兵；如果把它放开成 0，就必须再定义一个表示「无限」的返回值。
9. **全整数比较**：所有量都是 int 字节，判题用精确相等而不是容差；`static` 只是三项相加，别把
   `cuda_overhead` 也并进去（它属于峰值但不属于训练态）。
10. **分工**：字节数本身在 `transformer-mem` / `training-state-memory`，分片后每卡多少在
    `distributed-zero-shard-ownership`，offload 换多少带宽在 `distributed-zero-offload`；本题只管
    **给定预算反解边界**。

## 对拍口径

参考解：`max_micro_batch` 用整数 floor 除法、`max_seq` 用二次方程正根 + `while fits(...)` 前后校正、
`world` 用 `(G + d - 1)//d` 的 ceil 技巧、`global_batch_fits` 用**回乘相等**判据。判题 `_oracle` 全部换路：
`static` 用 `sum([...])`、峰值把 $m$ 提到括号外、`headroom` 一路连减、两条边界都用**纯整数二分**
（上界取「只算线性项」得到的 `budget // (s*ap) + 1`，它一定是合法上界，全程不碰浮点与开方）、
`world` 用「商 + 余数进位」、`global_batch_fits` 用**取模**判据。另设第三方裁判：测试里独立把
$\text{peak}(m,s)$ 公式代回去，逐条断言 `bound` 放得下而 `bound+1` 放不下；再叠加 1 字节阶梯
（预算 $k\cdot c$ 时边界恰为 $k$、少 1 字节就掉到 $k-1$）、$m\leftrightarrow s$ 不对称、闭区间边界、
reserve 与 overhead 的归属、负 headroom、$G$ 不整除等退化用例。

## 复杂度

参考解 $O(\log \text{budget})$（二次根的 ±1 校正在浮点误差下至多一步）；判题的整数二分是
$O(\log(\text{budget}/ap))$，整套用例在毫秒级完成。
