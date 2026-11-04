# 扩散：SeaCache 频域指示器门控步（SEA 滤波 + 累积预算 + 残差外推）

## 题意

SeaCache（Spectral-Evolution-Aware Cache）用**频域指示器**判断视频扩散相邻步变化是否足够小：对原始视觉隐张量做三轴 Wiener 型频域滤波得到 SEA 指示器，相邻步指示器的**累积相对 L1** 不超预算就跳过块栈。实现 `solve(vision_latent, gen_input, gen_output, state, step_index, sigma, num_inference_steps, threshold=0.25, residual_order=1, retention_steps=1, cache_end_steps=1, max_consecutive_cached=2, power_exp=3.0)`，每步三段：

```text
# 1. SEA 滤波指示器 (vision_latent (C,T,H,W) -> movedim(0,-1) -> (T,H,W,C))
spectrum = fftn(x, dims=(0,1,2));  sigma <- clamp(sigma, 1e-6, 1-1e-6)
clean_power = 1/(|f|^power_exp + 1e-16)                    # 每空间轴
axis_gain   = (1-σ)·cp / ((1-σ)²·cp + σ² + 1e-16)
gain = Π axis_gain;  gain <- gain / mean(gain)             # 密度归一
indicator = ifftn(spectrum·gain).real

# 2. 门控 (先处理步邻接, 再判预算)
非相邻 (step ≠ 上次+1) -> 整条轨迹重置
distance  = mean|cur − prev| / (mean|prev| + 1e-16)        # 形状/dtype 不符 -> inf
candidate = accumulated_distance + distance
compute   <=>  强制 ∨ candidate >= threshold               # 注意是 >=
强制 (or): step < retention_steps、step >= N − cache_end_steps、
           previous_indicator is None、consecutive >= max_consecutive_cached(>0 时)
compute 步 accumulated 归零; skip 步 accumulated = candidate

# 3. 残差复用
compute 步: output = gen_output; history 追加 (step, gen_output − gen_input)
            并截断到 residual_order+1; consecutive = 0
skip 步:    output = gen_input + cached_residual; consecutive += 1
            residual_order=1 且有两条不同步记录时线性外推:
            cached = latest + (latest − previous)·(step−latest_s)/(latest_s−prev_s)
无 history 却判 skip / history 残差形状或 dtype 与 gen_input 不符
            -> fail-open compute 且 accumulated 归零
```

返回 `dict(output, computed, distance, accumulated_distance, state)`，state 五键 `{history, previous_indicator, accumulated_distance, consecutive_cached, last_step}`（新 dict，纯函数式）。

## 关键语义

- **指示器在频域构造**：三轴增益连乘后按均值归一，`ifftn(...).real` 取实部回原 dtype；σ 在滤波内 clamp 到 `[1e-6, 1−1e-6]`（端点行为：T=H=W=1 时归一化增益恒 1，指示器≈恒等）。
- **`>=` 不是 `>`**：candidate 恰等于 threshold 即 compute——threshold=0 时 SeaCache 永不跳过（零距离 candidate=0 ≥ 0）。
- **distance 的参照是上一步指示器**（filtered），不是原始隐张量；分母是 `mean|prev| + 1e-16`。
- **非相邻步**（前跳或回跳，凡 ≠ 上次+1）先重置 history/previous_indicator/accumulated/consecutive——重置后首观测强制 compute。
- **外推 scale 可 ≠ 1**：history 两步不相邻时（如步 0、2）`step_scale = (3−2)/(2−0) = 0.5`。
- **fail-open 双通道**：门控判 skip 但 history 空、或 history 残差形状/dtype 不符，都回退 compute 并把 accumulated 归零。

## 判题域约定

- 官方 gate 有 (step, sigma) 备忘（root+leader 双入口同 forward）；判题域每步显式调用一次，备忘省略。
- 官方 raw_vision 是 latent 列表、distance 逐项平均；判题域单 latent。
- 官方 und 双流（Cosmos）、FSDP/并行同步、autograd/无 context 等 fail-open 分支不入实现；取 generic 视频 leader 路径。
- `threshold ≥ 0`、`residual_order ∈ {0,1}`、`retention_steps/cache_end_steps/max_consecutive_cached ≥ 0`、`sigma ∈ [0,1]`、`power_exp > 0`（sea_filter 内校验）为判题域显式校验。

## 例子

6 步、retention=1、cache_end=1、max=2、threshold=0.25：步 0 强制 compute；步 1 微变（distance≈0.01）skip；步 2 大变（candidate≈0.3）compute+归零；步 3 同步 skip；步 4 微变 skip（consecutive=2）；步 5 触 cache_end 窗强制 compute。computed 序列 `[T,F,T,F,F,T]`。

数据规模：latent `(1–2, 1–2, 1–4, 1–4)` float32，N ≤ 8，threshold ∈ [0, 10]。
