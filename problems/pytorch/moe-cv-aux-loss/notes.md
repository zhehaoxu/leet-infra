# 解题说明 · CV 辅助损失

## 与 Switch aux loss 的取舍

- Switch：$E \cdot \sum f_e P_e$，最小值 $= k$（当 perfectly balanced）；量纲跟 $E$ 相关。
- CV：$\mathrm{CV}^2$，最小值 $\approx 0$（当 perfectly balanced）；**无量纲**、$E$ 无关。

CV 更适合「不同专家数模型共用一套超参」；Switch 更适合单模型调优。

## 为什么 importance + load 都要惩罚

只惩罚 load：`load` 是 0 阶梯度不可微。只惩罚 importance：软分布可能均匀但硬分派仍偏。两者平方和把「软的可微信号」与「硬的实际负载信号」都注入梯度。

## std 用有偏 (unbiased=False)

OLNN 论文与常见实现均用 $\sqrt{\frac{1}{E}\sum (x - \bar{x})^2}$；有偏版让 $E=1$ 时 CV=0，与「单专家必然均衡」直觉一致。

## eps=1e-6

分母保护：`load.mean()` 与 `importance.mean()` 恒正（$kT/E$、$T/E$），eps 只在 $T=0$ 或 $k=0$ 才起作用。保留是为了和主流实现完全对齐。
