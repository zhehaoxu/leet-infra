# 详解：SGDR 余弦暖重启与周期账本

## 出处
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, arXiv:1608.03983
  （式 (3)：\(\eta_t=\eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos(\frac{T_{cur}}{T_i}\pi))\)，
  并说明"每次重启后周期长度乘 \(T_{mult}\)"；\(\eta_{max}\) 取该组初始 lr）。
- 参考实现：PyTorch 原生 `torch/optim/lr_scheduler.py::CosineAnnealingWarmRestarts`
  （`__init__` 的 `T_i=T_0`/`T_cur=last_epoch`、`get_lr` 的余弦式、`step()` 中
  `T_cur+=1` 后紧跟 `if T_cur >= T_i: T_cur %= T_i; T_i *= T_mult` 的顺序）。
  原生在构造时会自动走一次 `step()`（`last_epoch=-1` ⇒ 落到 \(T_{cur}=0\)、lr=base_lr），
  与本题"构造即第 0 步"的约定天然对齐，故可逐位对拍。
- todos.md #90④ 给的 arXiv:1608.03983 是正确的（已回源核对论文原文与原生实现一致）。

## 核心与易错点
1. **先重启、后写 lr**：`T_cur` 加到等于 `T_i` 时立刻取模归零并把 `T_i` 乘 `T_mult`，然后才用**新**的
   \((T_{cur},T_i)\) 算 lr。因此 \(T_{cur}/T_i\) 永远取不到 1，**谷底 \(\eta_{min}\) 永不被输出**；
   若把顺序写成"先写 lr 再重启"，序列会在每个周期末端多出一个 \(\eta_{min}\)，且重启点后移一步。
2. **\(T_{cur}\bmod T_i\) 而不是置 0**：`T_mult=1` 时每 \(T_0\) 步重启，增量版等价于置 0；
   但如果实现成"整除定位"却忘了同步 `T_i`，\(T_{mult}>1\) 时就会全线错位。周期长度是**累乘**关系。
3. **重启点不是 \(T_0\) 的倍数**：\(T_{mult}=2,T_0=3\) 时重启在 \(t=0,3,9,21,\dots\)（间隔 \(3,6,12,24\)），
   即 \(T_0(T_{mult}^n-1)/(T_{mult}-1)\)。评测的重启点账本就是卡这一条的。
4. **构造即第 0 步**：\(lr(0)=\) base_lr（周期峰值），别多走一步把 \(t=0\) 读成 \(T_{cur}=1\)。
5. **\(T_{mult}=1\) 是合法退化**：周期恒定，\(T_i\) 不变；此时重启点为 \(T_0\) 的整数倍。
   \(T_0=1\) 也是合法退化：每步都重启 ⇒ lr 恒为 base_lr。
6. **无终点、无夹紧**：与 OneCycle/WSD 不同，本调度器不接收总步数，读得再远也只是继续拉长周期，
   不会钉死在 \(\eta_{min}\)；`eta_min` 只是下界。
7. \(T_i\) 增长是**整数累乘**，步数很多时 \(T_{cur}/T_i\) 仍要用 float 除法（原生是 `math.pi * T_cur / T_i`）。

## 对拍口径
- 外部参照：**原生 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`**，同一个 `SGD(lr=base_lr)` 起头、
  同一约定读 \(t=0..29\)，逐点绝对误差 < 1e-12（7 组配置：\((T_0,T_{mult})\) 取 \((3,2),(4,1),(5,2),(2,3),(1,2),(6,2),(7,1)\)，
  含 `eta_min` 非零与 base_lr=1.0）。
- 独立重算：`_oracle` 走闭式路线——用整数累加 \(S_n=\sum_{k<n}T_0T_{mult}^k\) 定位第 \(t\) 步落在哪个周期、
  \(T_{cur}=t-S_n\)，完全不模拟递推状态；容差 1e-12，跑 60 步（跨 4 个周期）。
- 示例数值由参考解实跑得到（并与原生输出逐位相同）。
- 退化/不变量：重启点账本恰为 \([0,4,12,28]\) 且间隔 \(=\{T_0T_{mult}^i\}\)；\(\min(lr)>\eta_{min}\) 且每个周期内严格单调下降；
  \(T_{mult}=1\) 时重启点为 \(T_0\) 整数倍、谷底值 0.015278640450004212 精确可复现。

## 复杂度
每步 \(O(\text{param groups})\) 的一次余弦计算加常数状态更新；周期数 \(n\) 随步数对数增长
（\(n\approx\log_{T_{mult}}(t/T_0)\)），\(T_i\) 为整数累乘，不随步数线性膨胀。
