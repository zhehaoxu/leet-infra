# 解题笔记：TCD gamma 两段式一致性步

## 回源定位

- 源文件：diffusers `schedulers/scheduling_tcd.py`（810 行，TCD 论文 arXiv 2410.05964，2024-10）。核心是 `step()`（L594-703）。
- `set_timesteps` 两条路径：默认（original_inference_steps None，k = num_train // original_steps 的等距抽取）与自定义 timesteps（**严格递减校验** `timesteps[i] >= timesteps[i-1]` 报错）。判题域只考 step 语义，表作为入参直接给。
- 官方 `step()` 里 eta 用 `assert`（AssertionError），消息 "gamma must be less than or equal to 1.0"——判题域统一 ValueError 但保留关键词（reject 用例断言子串）。

## 数学要点

1. **gamma 中间级是 TCD 的灵魂**。LCM/consistency 的多步采样每步只做「x_t → x0 → x_prev」；TCD 把重加噪的目标改成 gamma 插值级 `s = floor((1−eta)·prev_t)`：先 `x_s = √ᾱ_s·x0 + √(1−ᾱ_s)·ε`（**同一个 ε**，保持模型输出的噪声方向），再从 `x_s` 按 DDPM 边缘分布比例 `√(ᾱ_prev/ᾱ_s)` 加新噪声到 prev。eta=0 时 s=prev_t，两段合一（确定性 LCM 式）；eta=1 时 s=0，回到 x0 全重加噪。
2. **floor 语义**。`floor(7.5)=7` 而 round=8——专门用 eta=0.25、prev_t=10 的案例判别（round 变异在此被抓）。
3. **三参数化推导**。epsilon：`x0=(x−√β_t·ε)/√ᾱ_t`；sample：`ε=(x−√ᾱ_t·x0)/√β_t`；v_prediction：`x0=√ᾱ_t·x−√β_t·v`、`ε=√β_t·x+√ᾱ_t·v`（diffusers 的 v 定义是 `√ᾱ·ε−√β·x0`，即 get_velocity——合成互验时 mo 的 v 公式第一版写反成 `√β·x0−√ᾱ·ε`）。
4. **末步门控与 prev_t 取 0 的组合**。末步（step_index == len−1）prev_t 取 0 且不加噪；中间步的 prev_t 来自表。官方 `final_alpha_cumprod`（prev<0 分支）不可达——prev_t 越界取 0 而非 −1，死代码不入实现（docstring 记录）。

## 实现要点

- 校验顺序：mo/sample → ᾱ（张量/1D/长度≥2/浮点）→ timesteps（张量/1D/非空/整型/**索引范围先行**、严格递减在后）→ step_index → eta → prediction_type。索引范围校验必须排在递减之前，否则 `_tab(-1, 0)` 会先触发递减错误（第一版即踩此坑）。
- `noise_applied` 作为返回值显式暴露，测试可直接断言门控（而不是从 prev 是否变化间接推断）。
- eta=0 且 noise 非 None：忽略 noise（官方分支不走、采样不发生），prev == x_s 逐位相等。
- 校验失败消息判题域固定：递减用「严格递减」、范围用「索引范围」、eta 用官方 gamma 关键词。

## 测试要点

- N=5 全循环对拍（oracle 用 math.sqrt 独立实现，ᾱ 表传 python list）：每步断言 x0/ε/x_s/prev 四元组 + timestep_s/prev_timestep/noise_applied 元信息。
- **合成互验**：x = √ᾱ_t·x0 + √(1−ᾱ_t)·ε 逐列合成（两列不同的 (x0, ε) 对），三种参数化各自给 mo → x0/x_s 必须一致到 1e-12。**不能**用 `_t` 的线性构造 `[v, v·0.5−0.3]`——mo 的合成公式（如 v 的）对列是非线性的，线性构造会破坏逐元素关系（第一版即在此翻车）。
- floor 判别：eta=0.25、prev_t=10 → 7（round 给 8）；eta=0 → 恒等 10；eta=1 → 0。
- 末步换 noise 逐位相同（乘零语义的 TCD 版：直接不走噪声分支）。
- 单步表 `[49]`：step 0 即末步、prev_t=0、s=0。
- eta=0：timestep_s == prev_t、prev == x_s 逐位、noise_applied False。

## 变异清单（34 条，全部 CAUGHT，无等价）

gamma 插值/表定位 6 条：floor→round、eta 系数减半、s 用当前 t、prev_step_index+2、prev 用当前位、表尾越界取 1。参数化 5 条：epsilon 符号/分母互换、sample 分子系数互换、v_prediction 符号/系数互换。两段式 9 条：x_s 系数互换、x_s 用 prev 级、门控三向变异（末门删/eta 门松/全删）、ratio 倒置、noise 系数复制、prev 符号、ᾱ_t 用 prev。校验/消息 14 条：递减翻转（允许相邻相等）、递减删、范围删、整型删、ᾱ 长度松、step 上界紧 1、eta 下界删、eta 消息改、prediction_type 收紧、noise 必给删、noise 形状删、形状删、ᾱ/timesteps 类型删、step 消息改。

经验：纯确定性公式 + 显式 `noise_applied` 返回值，变异全 CAUGHT 且无等价；结构重点（s/prev_t 取位、两段系数、门控方向）各由 2-6 个用例交叉覆盖。教训同 761：类型校验各自要有专属 reject 用例（本轮一步到位）；合成互验用例必须逐列显式构造，不能依赖线性模式构造。
