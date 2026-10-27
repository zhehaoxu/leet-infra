# 详解：VAE latent 的 shift / scale 与先验校正

## 出处

SDXL（Podell et al., arXiv:2307.01952）用 `scaling_factor=0.13025` 把 VAE latent 归一到单位尺度；SD3 / Flux 改用 `scaling_factor=0.3611`、`shift_factor=0.1159`，并在潜空间统计上做线性校正；Waltersdorfer et al.（ICLR 2024）系统分析了 latent 分布与 \(N(0,1)\) 先验不匹配导致的「编码-解码不一致」，并给出线性白化的修正口径。

## 核心与易错点

1. **顺序不能换**：编码 \((x-\text{shift})\cdot\text{scale}\)、解码 \(z/\text{scale}+\text{shift}\)。把 shift 与 scale 的顺序对调，或者解码先加 shift 再除，都会让 \(x\) 偏一个 \(\text{shift}\cdot(\text{scale}-1)\) 量级——判题用 `scale=0.5, shift=0.25` 的单点数值直接卡住。
2. **per-channel 统计量在第 1 维**：\(\mu_c,\sigma_c\) 形状 \((C,)\)，广播时要 reshape 成 \((1,C,1,1)\)；直接按最后一维广播会把通道搞错。标量统计量等价于「每通道同一个值」。
3. **`scaling_factor` 必须为正**：它是尺度，负数虽然没有数学障碍但会翻转语义，接口直接拒绝。
4. **统计量成对出现**：只给 mean 不给 std 是无意义配置，必须报错（而不是用 std=1 兜底）。
5. **往返是最强断言**：任意 `scale/shift/mean/std` 下 `encode(decode(z))` 都应还原 \(z\)，这同时检验了「解码用乘 \(\sigma_c\)、编码用除 \(\sigma_c\)」的方向一致性。
6. **数值范围**：解码后 \(x\) 的量级是 \(1/\text{scale}\approx 5.5\) 倍，float32 仍然充足；但中间若用 float16 累计会有可见误差，所以内部统一 float64。

## 对拍口径

参考解 float64 计算、出口 float32。判题 `_oracle` 逐元素按 \((batch,channel)\) 索引取统计量、用 python float 重算两个方向，另加四组结构断言：示例数值（有/无统计量两版）、五种参数组合的往返、顺序单点值（`scale=0.5`、`shift=0.25`）与默认恒等、per-channel 的零点映射与通道内放大倍数。阈值 `allclose(atol=1e-6)`（往返放宽到 `1e-5`）。

## 复杂度

逐元素 \(O(N)\)，无归约（统计量由外部给定）。
