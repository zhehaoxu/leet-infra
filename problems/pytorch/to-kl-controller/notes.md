# 详解：PPO 自适应 KL 系数控制器

## 出处
Ziegler et al., *Fine-Tuning Language Models from Human Preferences Using RL*（arXiv:1909.08593）附录的自适应 KL 控制；HuggingFace TRL `AdaptiveKLController.update`：`proportional_error = clip(current/target - 1, -0.2, 0.2); mult = 1 + proportional_error * n_steps / horizon; value *= mult`。本题聚焦这一控制器本身。

## 核心与易错点
1. **是比例 `current/target - 1`，不是差 `current - target`**：控制器按相对偏差调节，量纲无关；直接用绝对差会在目标 KL 很小时调过头。
2. **误差先截断到 ±0.2 再乘 \(n/H\)**：顺序是 clip 在乘法之前，防止单周期暴涨；写成先乘后 clip 结果不同。
3. **\(n/H\) 是平滑因子**：步数越多、horizon 越小，调节越激进；\(n/H\to0\) 时几乎不变。
4. **方向**：KL 超标（\(d>d_\text{tgt}\)）抬高 \(\beta\) 加惩罚，KL 不足压低 \(\beta\)——符号反了会让系统发散。

## 对拍口径
参考解 `beta * (1 + clip(current_kl/target_kl - 1, -0.2, 0.2) * n_steps/horizon)`。判题 `_oracle` 用纯 Python `max(-0.2, min(0.2, ...))` 手算截断、不复用 torch.clip（结构不同、数值等价），`abs(got-exp) < 1e-5`。另设「\(d=d_\text{tgt}\) 系数不变」「超/欠调方向相反」「大偏差饱和（更大 KL 结果相同）」断言。

## 复杂度
\(O(1)\) 标量运算。
