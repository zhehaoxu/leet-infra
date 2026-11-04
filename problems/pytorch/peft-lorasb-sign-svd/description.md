# PEFT：LoRA-SB 符号梯度估计的 SVD 初始化

## 题目描述

**LoRA-SB**（arXiv:2411.19557，CERT-Lab）把 LoRA 初始化从启发式换成数据驱动：先在小样本（约 170 条）上做一次前向/反向，把累积梯度压缩成**符号估计** \(\widehat{\Delta W}=-\mathrm{lr}\cdot\mathrm{sign}(\nabla W)\)（一阶信息的最激进压缩：只留方向、幅值统一为 lr），再对这个估计做**随机化低秩 SVD**，把因子**不对称**地拆进 LoRA 两侧——\(B=V\)、\(A=(U\,\mathrm{diag}(s))^\top\)，使 \(B@A\) 重构符号更新。r_squared 模式进一步在 A、B 之间插入 \(r\times r\) 中间映射 \(M\)（初值 \(\mathrm{diag}(s)\)），**只训练 M**，SVD 因子全部冻结。

本题复刻官方实现（CERT-Lab/LoRA-SB）的确定性账：

- **符号更新**（`gradient_utils.py`）：\(\widehat{\Delta W}=-\mathrm{lr}\cdot\mathrm{sign}(g)\)，`sign(0)=0`，任意非零幅值坍缩到 \(\pm\mathrm{lr}\)。
- **svd_grad**（`svd_utils.py::get_svd_grad`）：官方调用传 \(X=\widehat{\Delta W}^\top\)（(in,out)）；float32 管线 + `manual_seed(seed)`；**在 `svd_lowrank` 之前先抽一个从未使用的 dummy `randn(n_cols, rank)`**——值没用但**推进了 RNG 流**，`svd_lowrank` 内部随机投影从流当前位置取数，删掉这行结果就变（怪癖）；返回 \(enc=U\,\mathrm{diag}(s)\ (d_\text{in},r)\)、\(dec=V^\top\ (r,d_\text{out})\)、\(s\)，出域前转 float64。
- **init_ab**（`initialization_utils.py`）：\(B_w=dec^\top\ (d_\text{out},r)\)、\(A_w=enc^\top\ (r,d_\text{in})\)；`half_init_dec` 对 **转置前** 的 dec 的**下半** \(\lceil r/2\rceil\) 行做 `kaiming_uniform_(a=√5)`（bound \(=1/\sqrt{d_\text{out}}\)，fan_in 看被初始化矩阵的列数）；`random_init` 用同一全局流**先 enc 后 dec** 整体重初始化（kaiming a=√5；enc (in,r) 的 fan_in=r）；half_init 先做、值被 random_init 覆盖但 **RNG 流痕迹保留**（两次抽样数不同 → 最终值不同）。
- **latent 账**（`latent_utils.py`）：r_squared 前向 \(B(M(A(x)))\cdot\text{scaling}\)（三段嵌套，\(M\) 一般非对称矩阵），与合并式 \(\Delta W=B@M@A\cdot\text{scaling}\) 数学等价；grad 模式 \(M=\mathrm{diag}(s)\)。
- **非 r_squared 回退**：\(A\sim\mathcal N(0,10^{-5})\) 近零初始化，只训 B。

## 任务

- `mode="sign_update"`：grad (out,in) 与 lr 入参，返回 \(-\mathrm{lr}\cdot\mathrm{sign}(grad)\)。
- `mode="svd_grad"`：X (in,out)、rank、niter、seed 入参，返回 (enc, dec, s)（float64）。
- `mode="svd_grad_nodummy"`：同上但**不含** dummy RNG 抽样（对照路径，非官方流）。
- `mode="init_ab"`：dec、enc 及 half_init_dec / random_init / seed 入参，返回 (a_w, b_w)。
- `mode="latent_forward"` / `mode="delta_weight"`：如上式。
- `mode="latent_init"`：s 入参，返回 \(\mathrm{diag}(s)\)。
- `mode="a_near_zero"`：r、in_features、seed 入参，normal_(0, 1e-5) RNG 重放账。
- 全程 float64 进出；SVD 管线内部按官方 float32。

## 示例

\(d_\text{out}=6, d_\text{in}=8, r=3\)，grad 为 randn(6,8)（seed 0），lr=0.01：

```
sign_update(grad, 0.01) -> (6,8)，非零元 = ∓0.01，零元 = 0
svd_grad(X, 3, niter=10, seed=42) -> enc:(8,3), dec:(3,6), s:(3,)
  s = [0.04752158001065254, 0.02828427217900753, 0.028284268453717232]（float32 管线逐位锚定）
init_ab(dec, enc, random_init=True, seed=7) -> a_w:(3,8), b_w:(6,4)
latent_forward(x, a_w, M, b_w, 2.0) == x @ delta_weight(b_w, M, a_w, 2.0).T
```

同 seed 下 `svd_grad` 与 `svd_grad_nodummy` 的 s **不相等**——dummy 抽样推进 RNG 流的铁证。

## 约束

- `svd_lowrank` 用官方 q=rank、niter 入参；rank<=0 抛 ValueError。
- `kaiming_uniform_` 无 generator 参数，按官方全局流 + manual_seed 复现。
- M、A、B 全部显式入参的部分不重算 SVD。
