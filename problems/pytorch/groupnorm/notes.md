# GroupNorm 详解

## 定义

把通道分成 $G$ 组，每组内的通道一起归一化：

$$ \hat{x}=\frac{x-\mu_g}{\sqrt{\sigma_g^2+\epsilon}},\qquad y=\gamma x+\beta\ (\text{按通道}) $$

$\mu_g,\sigma_g$ 在每个样本、每个通道组内统计。

## 定位：BN 与 LN 的折中

- BN 沿 batch（跨样本）统计 → 依赖 batch 大小、小 batch 差；LN 沿特征（逐样本）→ 适合序列。
- **GN 逐样本、沿通道组**统计：与 batch 无关、小 batch/检测分割友好，是 Stable Diffusion、多数视觉大模型的选择。
- 特例统一视角：$G=1$ ≈ 对整图做 InstanceNorm 的全局版；$G=C$（每组一个通道）退化为 **InstanceNorm**。GN 是 BN 与 IN 之间的连续旋钮。

## reshape 技巧（最优雅实现）

$(B,C,H,W)\to(B,G, C/G\cdot H\cdot W)$ 后对最后一维求 $\mu,\sigma$（`dim=-1, keepdim`），归一再 reshape 回 $(B,C,H,W)$。$\gamma,\beta$ 形状 $(C,)$ reshape 成 $(1,C,1,1)$ 广播。

## 坑

- 别沿 batch 维统计——GN 完全不看 N。
- `C % G == 0` 才有意义；组数不同结果不同，按题给 G。
- eps 在开方内。

## 对拍参考解

`F.group_norm(x, num_groups, weight, bias, eps)`。
