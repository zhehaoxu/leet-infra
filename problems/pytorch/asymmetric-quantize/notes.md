# 非对称量化 详解

## 量化映射

对称量化假设数据关于 0 对称、浪费掉一半量程去覆盖不存在的一侧。非对称量化用**仿射**把整段 $[\min,\max]$ 平移到 $[q_{\min},q_{\max}]$：

$$ s=\frac{\max-\min}{q_{\max}-q_{\min}},\qquad z=\operatorname{round}\!\Big(\frac{-\min}{s}\Big) $$
$$ q=\operatorname{clamp}(\operatorname{round}(x/s)+z,\;q_{\min},\;q_{\max}),\qquad \hat x=(q-z)\,s $$

uint8 用 $[0,255]$（$q_{\max}-q_{\min}=255$）。$z$ 是「浮点 0 落在哪个整数格点」。

## 何时用它

- **ReLU 后激活全 $\ge0$**：对称版会把 255 个格点浪费近一半（负半轴空转），非对称把 $[0,\max]$ 铺满 $[0,255]$，精度显著提升。
- 分布明显偏斜（如带正 bias 的特征图）同理受益。
- 权重通常近似零均值，更常用对称（省一个 zero_point、矩阵乘不用处理零点偏移）。

## zero_point 必须 clamp

$z=\operatorname{round}(-\min/s)$ 在张量**全正**（$\min>0\Rightarrow z<0$）或**全负**（$\max<0\Rightarrow z>255$）时会越出 $[0,255]$。`torch.fake_quantize_per_tensor_affine` 直接要求 $z\in[q_{\min},q_{\max}]$ 否则报错。标准做法是把 $z$ clamp 进范围——代价是极端单边分布下 $\min$ 侧还原不再精确，但格点仍在合法域内。

## 量化噪声与全零退化

- 往返误差 $\le s/2$，$s$ 由量程决定：**离群值撑大 $[\min,\max]$ → $s$ 变大 → 全体精度下降**。缓解手段是 percentile/clipping（用 99.99% 分位而非真 max）或 per-channel。
- 常数张量 $\max=\min$ → $s=0$ → 除零，需特判原样返回。

## 与对称版的取舍

| | 对称 | 非对称 |
|--|------|--------|
| scale | absmax/127 | (max-min)/255 |
| zero_point | 恒 0 | 需存/算 |
| 适合 | 近零对称（权重） | 偏斜（激活） |
| 矩阵乘 | 干净 | 要处理零点交叉项 |

## 关联

symmetric-quantize 是 $z=0$ 特例；fake-quant 概念（STE 前向）两题共用；量化家族是端侧部署与推理提速的核心。
