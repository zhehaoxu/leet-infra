## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_ddpm_wuerstchen.py`（269 行，Apache-2.0 / MIT 混合头，Pablo Pernías 原作）。关键行：`_alpha_cumprod`（L121-129：scaler 三分支 L122-125、cosine 平方 + 归一化 L126-128、clamp L129）、`__init__` 的 `_init_alpha_cumprod`（L116）、`previous_timestep`（L255-269：`argmin(|timesteps − timestep[0]|)` 只看 t[0]、`index+1` 查后继、expand 回 batch）、`step`（L202-221：后验 mu L212、std L215、`(prev_t != 0).float()` 门控 L216）、`set_timesteps`（L148-170：`linspace(1.0, 0.0, N+1)`）、`add_noise`（L223-250，与 `diff-ddpm-forward` 重复，不进判题域）。
- 出处：Wuerstchen（arXiv 2306.00637，Pernías et al.）的两级级联——prior 级生成低分辨率语义压缩（CLIP 空间），decoder 级在其上重建高分辨率；两级各持一个调度器实例，`scaler` 是「时间轴扭曲」旋钮（>1 把采样步挤向高噪声端）。
- 公式族：DDPM 后验 p(x_{t−1}|x_t, x_0) 的 ε 参数化——与 `diff-ddpm-step` 同一公式族（见查重）。

## 规范化（判题域 vs 官方）

- **噪声显式化**：官方 `randn_tensor(mu.shape, generator, ...)`；改为显式入参 `noise`，每次调用必需。
- **表与 t 显式化**：官方 `self.timesteps`（`set_timesteps` 产出）与 `self._alpha_cumprod` 的内部状态；本题 `timesteps_table` 入参 + 纯函数 ᾱ。
- **batch 语义收紧**：官方只用 `timestep[0]` 查 prev_t（`self.timesteps - timestep[0]`）但 ᾱ_t 用完整 t 向量——混合 batch 时 prev_t 与各元素 ᾱ 脱钩，语义破坏。官方管线保证 batch 内 t 相同；判题域直接校验「batch 内全部相等」并拒绝混合（官方无此校验）。
- **表尾越界转拒绝**：官方 `index+1` 越界是 IndexError；判题域校验后抛 ValueError。
- **scaler / s 校验**：官方无（scaler ≤ 0 会 pow 出垃圾）；判题域要求正有限。
- **门控保留浮点比较**：`(prev_t != 0).float()` 逐字——prev_t 来自表，精确 0 才触发；无 epsilon 容差（变异高发点）。
- **ᾱ clamp 保留**：`clamp(0.0001, 0.9999)` 逐字——两端在 t=1 / t=0 精确触发，防 √(1−ᾱ) 除零与 ᾱ=1 的 α 退化。

## 题眼

1. **ᾱ 是解析函数不是查表**：与 `diff-ddpm-step` 的离散 ᾱ 表入参形成对照。s offset 与归一化分母 `_init_alpha_cumprod` 是同一 cos 表达式在 t=s 处的取值——变异 `m-ac-init`（分母丢 s）与 `m-ac-offset`（分子丢 s）方向相反、都要抓。
2. **scaler 三分支**：>1 时 t→1−(1−t)^scaler（把 t 推向 1，更噪）；<1 时 t^scaler（拉向 0）；=1 恒等。Wuerstchen「两级」的实质——prior/decoder 各自的时间轴扭曲。三分支各一条变异 + 恒等分支用闭式 ᾱ 钉死。
3. **clamp 两端都有观测点**：t=1 → ᾱ=0 → clamp 下界（`check_first_step` 精确断言 0.0001）；t=0 → ᾱ=1 → clamp 上界（`check_identity_step` 断言 0.9999）。上界还能用表 [1, 0, 0] 的 t=0 触发（argmin 平局取首 → index=1 → prev_t=0 → 同时钉门控与恒等步）。
4. **查表是 argmin 最近邻 + 1**：不是索引减一、不是插值；平局取首个（torch/numpy argmin 语义一致）。t 必须取表内值（或接近）——测试全部从表内取 t，保证 argmin 定位无歧义。
5. **门控是浮点精确比较**：prev_t == 0 才关噪声；末步换 noise 输出不变是独立断言（`check_last_step_gate`），`m-gate` 与 `m-noise-applied` 分别钉计算路径与标志位。
6. **α = ᾱ_t/ᾱ_prev < 1**：表降序 → prev_t < t → ᾱ_prev > ᾱ_t。ᾱ 相等时（相邻表值映射同 ᾱ）恒等步——判题域允许（std=0、prev=sample），无需特殊处理。
7. **无状态机**：纯函数，全循环只是把表逐位走一遍——但每步 prev_t 查表仍可能错位（`m-idx-prev`），全循环用「prev_t 恰是表中下一个值」独立断言钉死。

## 查重

- **`diff-ddpm-step`**：同一 DDPM 后验公式族（mu/std 形状相同）。但那边 ᾱ 从**离散表**显式入参、prev_t 与 x0 路径不同；本题 ᾱ 是**解析函数**（s offset / 归一化 / clamp / scaler 幂变换四重加工）+ argmin 查表 + 门控。机制增量在 ᾱ 的构造与 prev_t 定位，不在后验公式本身。
- **`diff-ddpm-forward`**：前向加噪 q(x_t|x_0)（本题 `add_noise` 与其重复，不进判题域）。
- **`diff-ddim-step`（若在库）/ `diff-ddpm-step` 的确定性变体**：门控后的末步 ≡ 确定性收尾，但只是末步单点行为，非整段 DDIM。
- **`diff-dpmpp-2m` / `diff-lms`（757）等 multistep 族**：多步 history 状态机；本题单步纯函数、无 state。
- **`diff-cosine-dpmsolver-sde-step`（759）**：同为「cosine 命名」但那是 σ 空间 EDM/VE 约定 + SDE 噪声 + multistep；本题是 β-space ᾱ 后验 + 门控噪声，两题的 ᾱ/σ 公式与噪声语义完全不同源。

## 测试与变异

11 用例：首步 t=1（clamp 下界精确断言 + mu/std/prev 闭式独立验证）、末步门控（prev≡mu + 换 noise 输出不变）、中间步（ᾱ 双闭式 + α<1 + mu 闭式）、恒等步（表 [1,0,0] 的 argmin 平局 + clamp 上界 + α=1）、scaler>1 / <1 / =1 三分支闭式、B=2 batch 广播 + 混合 batch 拒绝、N=4 全循环（prev_t 独立断言 + 末步门控）、表五查、timestep 五查、参数七查——reject 全部断言错误消息防「空抓」。

34 条变异 **33 CAUGHT + 1 等价**：

| 等价变异 | 实证 |
|---|---|
| `m-argmin-t0`（查表 `t[0]` → `t[-1]`） | 200 组随机配置（表长 2-11、scaler 三分支、随机 s、B 1-2、半数表尾 0）全输出逐位 0 差异。结构恒等式：batch 内 timestep 全相等是前置校验，故 t[-1] ≡ t[0] |

其余 33 条覆盖：表五查（长度/有限/单调/维度/dtype）、timestep 五查（维度/有限/上下界/混合 batch/dtype）、scaler 与 s 的范围与 bool、noise 必需与形状、sample 形状、scaler 两分支指数、归一化分母丢 s、clamp 下界/上界/整体移除、s offset 丢失、cosine 平方降幂、查表 index+1→+0、表尾越界校验移除、门控恒开、noise_applied 标志、prev_t 返回值、mu 的 α 因子/符号/ε 项、std 的 (1−ᾱ_t) 因子、α 比值交换、ᾱ_prev 用 t[0] 替换、view 形状。

## 踩坑

- **门控标志与计算路径要分开钉**：`m-noise-applied`（只改返回标志）与 `m-gate`（只改计算）互不覆盖——一个错标志的实现对拍恒绿（prev 照算），一个错门控的实现标志恒对。两条变异各自被不同的断言抓到。
- **clamp 上界的观测点不平凡**：ᾱ(t) < 1 对一切 t ∈ (0, 1] 成立，上界只在 t=0 精确触发；而 t=0 作为当前步要求表中 0 之后还有后继——只能用相邻相等的表（[1, 0, 0]，单调不增允许）让 argmin 平局取首构造出来。若测试只有严格递减表，上界 clamp 变异（0.9999→1.0）会在 ᾱ=1 处除零崩、或恒等步断言漏抓。
- **官方 argmin 只看 t[0]**：混合 batch 时官方 prev_t 共享而 ᾱ_t 各异——这不是可复刻的「特性」而是管线不变量的泄漏；判题域选择校验拒绝并记录（变异 `m-t-batch` 由 reject 用例抓住）。
- **`t**scaler` 与 `1−(1−t)**scaler` 的 float 幂**：t=0 且 scaler<1 时 0^0.5=0 合法；scaler 变换后 t' 仍在 [0,1]（幂函数单调），无需额外 clamp。
