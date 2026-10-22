# Inverted Dropout 详解

## 定义与「反相缩放」

训练时以概率 $p$ 随机置零元素，并把**保留的元素除以 $1-p$**：

$$ \tilde x_i=\frac{m_i}{1-p}x_i,\qquad m_i\sim\text{Bernoulli}(1-p) $$

这样单个被保留单位的**期望不变**（$\mathbb E[\tilde x_i]=x_i$），于是**推理时不需要任何缩放**，直接原值前向。这是现代框架的默认做法（"inverted / scalable dropout"），区别于原始论文「训练不缩放、推理乘 $1-p$」。

## train vs eval（核心分叉）

- `training=True`：采样掩码 + 除 $1-p$。
- `training=False`（或 $p=0$）：**恒等返回**，不采样不缩放。判错口径数值全乱。

## 可复现：独立 Generator 锁 seed（本关关键）

用**独立** `torch.Generator(device=x.device).manual_seed(seed)` 采样，而**不碰全局 RNG**：

```
g = torch.Generator(device=x.device).manual_seed(seed)
mask = torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype) >= p
```

要点：
1. `>= p` 保留（概率 $1-p$），别写 `> p` 或方向反。
2. `dtype=x.dtype`、`device=x.device` 必须跟随输入，否则半精度/设备不匹配。
3. 用独立生成器，测试才能对同一 seed 复现同掩码；若动全局 `torch.manual_seed` 会污染其它随机算子。

## 与其它正则

DropPath、SpatialDropout（整通道一起置零）、Variational Dropout（共享掩码跨时间步）是其结构化变体；核心都是「伯努利掩码 + 期望保持缩放」。
