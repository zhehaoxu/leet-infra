# 详解：YaRN 的波长 ramp 与注意力温度

## 出处

- YaRN：Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*，arXiv:2309.00071，§3.2 的 NTK-by-parts 插值（按波长分段）+ §3.3 的注意力温度 \(0.1\ln s+1\)。
- LongRoPE：microsoft/LongRoPE 用同一套 inv_freq 混合；transformers 的 `modeling_rope_utils._compute_yarn_parameters` 就是这份实现的官方移植，本题的三个档位一一对应它里面的 `yarn` / `linear` / `dynamic_ntk` 三个函数。

上游代码里的顺序是：先算 `pos_freqs = base ** (arange(0, dim, 2)/dim)`，再取 `inv_freq_extrapolation = 1/pos_freqs`、`inv_freq_interpolation = 1/(factor*pos_freqs)`；边界由 `find_correction_dim`（本题的 \(c(n)\)）给出，`linear_ramp_factor` 做完 clamp 后用 `inv_freq = interp*(1 - e) + extrap*e` 混合，其中 `e = 1 - ramp`。`attention_factor` 在 `LlamaRotaryEmbedding.forward` 里被同时乘到 cos 和 sin 上。

## 核心与易错点

1. **两个索引尺度不一样**：边界 \(low/high\) 用 \(d\) 算出来（`find_correction_dim` 里乘的是 `dim`），但 ramp 的自变量只铺 \(h=d/2\) 个维度。所以"外推区"实际覆盖了 \(i<low\) 的维度——如果误把 \(low/high\) 也除以 2，ramp 会整体左移，中间过渡带的位置全错。`check_ramp_structure` 直接用两组独立算出的 \(low/high\) 断言三段划分。
2. **`beta_fast` 管外推侧、`beta_slow` 管插值侧，方向容易反**：`low=floor(c(beta_fast))`、`high=ceil(c(beta_slow))`。因为 \(c(n)\) 随 \(n\) 递减、且 \(c(n)\) 前有整体正系数，`beta_fast > beta_slow` 时才有 \(low<high\)。把两个参数对调，ramp 的斜率变负，clamp 之后会得到一段反序的混合——本题直接用 `beta_fast <= beta_slow` 抛 `ValueError` 把这种配置挡在门外。
3. **`truncate` 是真实的数值差异，不是开关装饰**：取整把边界吸附到整数维上。示例 \(d=8\) 时 \(c(32)=1.008\)、\(c(1)=2.513\)，取整后 \(low=1,high=3\)，\(i=2\) 的 ramp 恰为 0.5，得到 `0.00625`；不取整时 \(low=1.008,high=2.513\)，同一个 \(i=2\) 的 ramp 变成 \(0.6594\)，结果是 `0.00505697`。差 20% 以上，一眼可查。
4. **温度不是旋转**：\(\gamma\) 同时乘在 cos 和 sin 上，于是 \(\|q\cdot\cos+\mathrm{rot}(q)\cdot\sin\|=\gamma\|q\|\)。数学上这等价于把 attention logits 整体乘 \(\gamma^2\)（因为 \(QK^\top\) 的两侧各带一次 \(\gamma\)），正是论文"补偿插值带来的熵下降"的意图。若只乘 cos 或只乘 sin，就既不是旋转也不是缩放——`check_cos_sin_amplitude` 用长度断言把这条钉死。
5. **\(\gamma\) 的三档优先级**：`attention_factor` 显式给出时直接用它；否则若 `mscale` 与 `mscale_all_dim` **都**给出，取两者 `get_mscale` 之比；否则用 \(g(\text{factor},1)\)。只给其中一个 mscale 时它必须被忽略——上游代码是 `if mscale and mscale_all_dim:`，注意这里还有"真值"语义：`mscale=0` 会走 else 分支，本题改为显式 `is not None` 判断，避免这个坑。
6. **`factor <= 1` 时 \(\gamma=1\)**：`get_mscale` 在 `scale <= 1` 时直接返回 1（不取对数），所以 `factor=1` 的 YaRN 退化成"整张 inv_freq 不变、没有温度"的普通 RoPE。`check_attention_scaling` 和 `check_linear_mode` 各断言一次。
7. **`dynamic_ntk` 的 base 重写用的是当前长度**：\(\text{base}'=\text{base}\cdot((\text{factor}\ell/L)-(\text{factor}-1))^{d/(d-2)}\)，\(\ell\) 先被抬到至少 \(L\)（不足训练长度时 `base'=base`，指数再大也没用）。指数是 \(d/(d-2)\) 而不是 \(d/(d-2)\) 的某种近似——\(d=2\) 会除零，所以 `head_dim` 下限取 4。
8. **`linear` 与 YaRN 不是同一个东西**：`linear` 把**所有**维度都除以 factor（包括高频维），而 YaRN 只压低频维。两者的 `inv_freq[0]` 在 `linear` 下是 `1/factor`、在 YaRN 下仍是 `1`。如果只测 `linear`，任何"整张除以 factor"的错误实现都会通过；所以本题额外断言 ramp 两端分别贴合 `inv_extrap` 与 `inv_interp`。
9. **ramp 的奇异保护**：`low == high` 时上游给 `max += 0.001`。这种情况出现在两个边界反解到同一维（极端 `beta` 或很小的 `d`），不加保护会 0/0 出 NaN。本题保留该分支并在约束里把 `beta_fast > beta_slow` 作为前置校验，因此只能通过极端 `d`/`L` 触发。

## 对拍口径

参考解全部在 float64 下算 `inv_freq`、\(\gamma\) 与 cos/sin，最后统一转 float32。判题有三层：

- **示例值**：\(d=8\) 的那组 `inv_freq` / \(\gamma\) / `cos(pos=3)` / `sin(pos=3)` 六位小数硬编码；
- **结构断言**：按独立算出的 \(low/high\) 检查外推区、插值区与中间带（`interp < inv < extrap`）；检查 `inv_freq` 单调不增；检查 `truncate` 开关带来的数值差、`beta_fast` 变大时过渡带左移；检查 \(\gamma\) 的四种来源（默认 / factor=1 / mscale 之比 / 显式 `attention_factor`）；检查"施加 cos/sin 后向量长度变为 \(\gamma\) 倍"；检查 `linear` 与 `dynamic_ntk` 的退化恒等式（`seq_len < L` 被抬到 \(L\)、未超长时 `base` 不变）；
- **双 oracle**：① 独立 numpy `_oracle`（标量公式重算 `inv_freq`、`γ`、`cos`、`sin`）对 5 组 `yarn` 配置 + `linear` + `dynamic_ntk` 逐元素对拍，阈值 `1e-6`；② 直接对拍 `transformers==5.20.2` 的 `LlamaRotaryEmbedding`（`rope_parameters={"rope_type": "yarn", ...}`）与 `modeling_rope_utils._compute_dynamic_ntk_parameters`，`inv_freq` 用 `1e-5`、cos/sin 用 `1e-4`（上游在 float32 里算 `pos·inv_freq`，位置到几百时相位误差已到 \(10^{-5}\) 量级，因此 cos/sin 的容差比 `inv_freq` 松一档）。

另有 27 组非法输入必须以 `ValueError` 拒绝：奇数/过小的 `head_dim`、非整数或 `bool`、`base<=1`、`factor<=0`、`L<2`、`beta_fast<=beta_slow`、`truncate` 非 bool、未知 `rope_type`、`position_ids` 非张量/非二维/非整数/含负/为空、`linear` 下传 `attention_factor` 或 `mscale`、\(\gamma\) 非正、`dynamic_ntk` 的 `seq_len` 非正整数。

## 复杂度

`inv_freq` 与 cos/sin 各是 \(O(d)\) 与 \(O(BSd)\) 的逐元素运算，边界反解是常数次 `log`/`floor`/`ceil`：时间 \(O(BSd)\)，空间 \(O(BSd)\)（cos、sin 各一份）。
