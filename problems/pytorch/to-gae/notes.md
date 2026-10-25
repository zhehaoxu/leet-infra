# 详解：广义优势估计 GAE

## 出处
Schulman, Moritz, Levine, Jordan, Abbeel, *High-Dimensional Continuous Control Using Generalized Advantage Estimation*（arXiv:1506.02438）。GAE 定义为 \(\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V\)，实现上用反向递推 \(\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\) 一次扫完，是 PPO 训练里优势的标准算法。

## 核心与易错点
1. **反向递推**：必须从 \(t=T-1\) 倒着算，`last = delta + gamma*lam*last`；正向累加会错。
2. **TD 误差用 V_{t+1}**：\(\delta_t=r_t+\gamma V_{t+1}-V_t\)，末端 \(V_T=0\)（不 bootstrap 终止后价值）。越界索引要判 `t+1<T`。
3. **return = advantage + value**：价值回归目标 \(R_t=\hat A_t+V_t\)，不是 \(\hat A_t\) 本身。
4. **λ 的两端**：λ=0→单步 TD（低方差高偏差）；λ=1(γ=1)→蒙特卡洛（无偏高方差）。中间权衡。

## 对拍口径
参考解反向递推 advantages，returns=advantages+values。判题 `_oracle` 先物化整个 `delta` 列表，再倒序 `acc=delta[t]+gamma*lam*acc`（结构与参考解不同），两个张量分别 `allclose(atol=1e-5,rtol=1e-5)`。另设 λ=0 退化为单步 TD、returns-advantages==values 恒等、T=1 边界。

## 复杂度
\(O(T)\) 单次反向扫描。
