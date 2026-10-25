# 详解：指数滑动平均（EMA）权重与升温

## 出处与回源
- **EMA 影子权重**思想见 Mean Teacher（Tarvainen & Valpola, arXiv:1703.01780）：对权重做时间维指数滑动平均得到更稳定的目标模型；扩散模型（如 DDPM/EDM）、Model Soup 评测平均也广泛使用。
- **升温式 \(d_n=\min\!\left(decay,\frac{1+n}{10+n}\right)\)**：核实自 llava 的 `model/ema.py`（继承自 MAE / `timm` 的 `EMAHelper`）逐字实现——原文注释即「`decay = min(decay, (1 + num_updates) / (10 + num_updates))`」。含义：更新数很少时令 \(d\) 偏小、影子快速跟上当前权重，避免被随机初始化「钉死」；\(n\) 大了 \(d\) 才封顶到 `decay`。

## 核心思想
\[shadow \leftarrow d\cdot shadow+(1-d)\cdot param\]
\(d\) 越接近 1，影子越「懒」、越平滑；升温让 \(d\) 从低爬升。本题把「按更新数升温」与「定值 decay」两条路径统一在一个 \(d_n\) 里。

## 为什么容易写对不上（易错点）
1. **\(n\) 的基准与递增时机**：构造时 \(n=0\)、\(shadow=p\)（不动）；**每次 `update()` 先 \(n\leftarrow n+1\) 再算 \(d_n\)**。所以第一次 `update()` 用的是 \(n=1\)（\(d=2/11\)），不是 \(n=0\)。
2. **影子是 clone，不是引用**：`shadow` 必须是与 `param` 分开的张量；若写成同一内存，`update()` 会把 param 一起改掉、且 \(n=1\) 起 param 被污染。
3. **就地且只改影子**：`update()` 不修改 `params`（不 `zero_grad`、不写回 param）。读的是 `params` 的**当前值**——评测会在两次 `update()` 之间就地改动 `params`。
4. **升温封顶**：\(\min(decay,\cdot)\) 而非直接 \((1+n)/(10+n)\)；后者 \(n\to\infty\) 会趋近 1 超过给定 `decay`。`warmup=False` 时恒为 `decay`。
5. **多参数共享 \(d_n\)**：一次 `update()` 对所有影子用同一个标量 \(d_n\)，不是每个参数各自的计数。

## 对拍口径
判题用独立重算 `_oracle_ema`（把整条参数轨迹喂进一个纯 Python 循环、用 `d*n + (1-d)*p` 的非就地写法），与参考解 `allclose(atol=1e-5, rtol=1e-5)` 比对。示例完全确定（无随机数）。检查覆盖：
- `check_example`：升温、单参数 7 次 `update()` 的具体影子值。
- `check_matches_oracle_multi`：多参数、混合形状、逐步改 param，影子与 `_oracle_ema` 对拍。
- `check_no_warmup_fixed_decay`：`warmup=False`、`decay=0.5`，验证每步恒 \(d=0.5\)。
- `check_decay_zero_tracks_exactly`：`warmup=False`、`decay=0.0` → 影子每次都等于当前 param。
- `check_shadow_independent_and_params_untouched`：`update()` 后 `params` 值不变、且 `shadow` 与 `param` 非同一张量。

## 复杂度
每次 `update()` \(O(\text{numel})\)；额外一份额外的影子权重（约与模型等大的显存）。
