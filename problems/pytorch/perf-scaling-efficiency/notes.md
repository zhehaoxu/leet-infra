# 详解：强/弱扩展的加速比、效率与饱和点

## 出处

Megatron-LM（Shoeybi et al., arXiv:1909.08053）与 MLPerf Training 的扩展曲线都用同一对口径：
**strong scaling** 固定总问题规模、看 step 时间随卡数下降多少；**weak scaling** 固定每卡负载
（如每卡固定 micro-batch）、看 step 时间是否保持不变。超线性点见 Sun 等的 scaling-of-training
（arXiv:2201.11854，小模型分片后通信占比骤降会出现 $E>1$）。强/弱扩展的定义可追溯到
Dowdy & Foster 1982 的经典表述。

## 核心与易错点

1. **两种模式的分子不一样**：strong 每步总样本数固定，整机吞吐只看 $1/t_i$；weak 每卡样本固定，
   每步总样本正比 $w_i$，吞吐是 $w_i/t_i$。把 strong 也按 $w/t$ 算，效率会被抬高 $w_i/w_0$ 倍
   ——16 卡时虚高 16 倍，是最典型的错。
2. **效率只除以卡数倍数**：$\text{efficiency}=\text{speedup}/(w_i/w_0)$ 对两种模式都成立；展开后 strong 是
   $(t_0/t_i)(w_0/w_i)$、weak 是 $t_0/t_i$。别再对 weak 的效率除一次 $w$。
3. **基线不一定是 1 卡**：所有比值都除以第 0 档，`world_sizes=[2,8,32]` 这种基线是 2 卡的情况要按 $w_0$ 归一。
4. **超线性是严格大于**：$\text{efficiency}=1.0$（示例里 4 卡那档 $100/25/4=1$）不算超线性点。
5. **饱和点含「持平」且取第一处**：判据是 $\text{throughput}_i\le\text{throughput}_{i-1}$，$t_i=t_{i-1}$ 这种
   「加卡白加」要算饱和；整条曲线严格上升时返回 `-1`（不是最后一档的下标）。
6. **自证恒等式**：$\text{speedup}_0=\text{efficiency}_0=1$，且逐点满足
   $\text{speedup}_i=\text{efficiency}_i\cdot(w_i/w_0)$——判题拿它做结构性断言。
7. **完美扩展的形状**：完美 strong $\Rightarrow t_i=t_0 w_0/w_i$、$\text{speedup}=w_i/w_0$、效率恒 1；
   完美 weak $\Rightarrow t_i=t_0$、效率恒 1。两条退化曲线各有一条用例钉住。

## 对拍口径

参考解严格按「整机吞吐」定义生成 `throughput` 再取比值；判题 `_oracle` 走闭式（strong：
$t_0/t_i$ 与 $(t_0 w_0)/(t_i w_i)$，一次除法算完），并把两个离散判据换成**整数交叉相乘**以避开浮点除法在
持平点上的舍入：超线性 $\iff t_0 w_0>t_i w_i$（strong）/ $t_0>t_i$（weak），饱和 $\iff t_i\ge t_{i-1}$
（strong）/ $w_i t_{i-1}\le w_{i-1}t_i$（weak）。另设用例：完美强/弱扩展、加卡持平即饱和、单点退化、
$\text{speedup}=\text{efficiency}\cdot w/w_0$ 恒等、两模式效率之间的 $w_0/w$ 关系、superlinear 集合与
efficiency 张量自洽。

## 复杂度

$O(n)$，$n$ 为档位个数（$\le 1024$）；无额外分配（除返回的三个张量）。
