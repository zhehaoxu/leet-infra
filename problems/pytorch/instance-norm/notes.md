# InstanceNorm（前向）详解

## 定义

对**每个样本、每个通道单独**在空间维上归一化：

$$ \hat{x}_{n,c}=\frac{x_{n,c}-\mu_{n,c}}{\sqrt{\sigma_{n,c}^2+\epsilon}}\cdot\gamma_c+\beta_c $$

统计只在 $(H,W)$ 上做，跨样本、跨通道都不共享。

## 用在哪

- 风格迁移（AdaIN 核心就是逐通道 IN 后对齐均值方差）、GAN 生成器。IN 去掉每张图各自的对比度/亮度统计，适合「内容—风格」解耦。
- 与 GN 关系：IN 等价于 GN 中 $G=C$（每通道自成一组）。

## 维度与参数坑

- 对 $(N,C,H,W)$：$\mu,\sigma$ 形状 $(N,C)$，归约 dim=(2,3)，reshape $(N,C,1,1)$ 广播。
- `affine=True` 时 $\gamma,\beta$ 形状 $(C,)$（按通道共享，不随样本变）；`track_running_stats` 一般为 False（IN 通常不维护 running stats，训练/推理都是逐样本现算）。

## 对拍参考解

`F.instance_norm(x, running_mean, running_var, weight, bias, use_input_stats=True, momentum, eps)`——注意 IN 恒用输入自身统计。
