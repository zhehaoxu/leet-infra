## 回源

- `huggingface/diffusers` 的 `src/diffusers/schedulers/scheduling_k_dpm_2_ancestral_discrete.py`（750 行，Apache-2.0）。关键行：`set_timesteps` 的 up/down/interpol 三表构造（L323-336）与四张 repeat_interleave 展开表（L338-341）；`step` 的首调/次调分支（L620-678）；`state_in_first_order = self.sample is None`（L528-538）；`gamma = 0`（L637-639，`sigma_hat = sigma * (gamma + 1)` 恒等于 sigma）。
- 论文：Karras et al. 2022（KDPM2 采样即 DPM-Solver-2 + ancestral 噪声注入），arXiv 2306.14898 是 ancestral 扩散的标准引用（Ho et al. 2020 DDPM 的 ancestral 采样）。
- 与 754（`diff-kdpm2-interleaved-step`，确定性版）同源同骨架，本题只考 ancestral 特有的三处机制差。

## 规范化（判题域 vs 官方）

- **噪声显式化**：官方 `randn_tensor(generator)`；无 generator 语义可钉，按全库纪律改为显式入参 `noise`（次调必需、形状校验），首调忽略。
- **状态外置**：官方 `self.sample / self.dt`；本题 `state` 只含 `sample`（`self.dt` 官方存了从不读——次调分支重算 `dt = sigma_down - sigma_hat`，死状态，不收）。`state_in_first_order = self.sample is None` → `state is None`。
- **表构造内联**：官方 `set_timesteps` 一次性构造；本题每次调用重算（N 小，纯函数化）。
- 保留官方逐字公式（含 nan 路径）：`up[N]` 的 0/0 = nan 不修（该格不被读）、`down[N]` 与 `interpol[N−1]/[N]` 按官方清场。

## 题眼

1. **三处机制差**（与 754 一一对应，直接出变异）：中点插值对象 σ_down（vs σ_next）、次调 dt = σ_down − σ（vs σ_next − σ 整步长）、落地注入 noise·σ_up（vs 无噪声）。
2. **数学冗余**：down[k] = σ_next[k]²/σ[k]、interpol[k] = σ_next[k]（精确恒等）——官方显式构造 interpol 是冗余的，但 **float32 下不冗余**（见下）。
3. **nan 清场**：torch 的 `lerp(有限, −inf, 0.5) = nan`（实测！不是 −inf 也不是凸组合），所以 k=N−1 与 k=N **两格**都是 nan，官方 `[-2:] = 0.0` 一格不能少；`down[-1] = 0.0` 清的是 k=N 的 nan（up[N] = nan 传播）。
4. **首调死赋值**：官方首调取出 `sigmas_up[i]`、`sigmas_down[i−1]` 但首调分支不读；i=0 时 `down[−1]` 负索引到尾格也不炸。
5. **末逻辑步无噪声**：σ_next = 0 → up = 0，up²+down² = σ_next² 的方差守恒使 ancestral 在末步自动退化为确定性一步。

## 数值陷阱：灾难性消去（官方真实缺陷）

`down[k] = √(σ_next[k]² − σ_up[k]²)` 中两项几乎相等（σ ≫ σ_next 时 up ≈ σ_next），差是 σ_next²·(σ_next/σ)²——**相对误差被放大 (σ/σ_next)² 倍**。float64 域（本题）无害；float32 域（官方 pipeline）实测 [7.90, 0.205, 0] 表 down[0] 相对误差 **4.2e-5**（exact 0.005306328 vs float32 算得 0.005306550）。稳定形式是 `σ_next²/σ`，官方逐字公式保留（判据），notes 记录缺陷。连锁效应：interpol 中点数学上 = σ_next，float32 下因 down 的消去误差而**不等于** σ_next——所以「官方显式构造 interpol」在 float32 下是**有行为差异的**，不可用 σ_next 替换（对应变异 `m-lerp-to-next` 被抓）。

## 查重

- **754 `diff-kdpm2-interleaved-step`**：同骨架，五点差异——插值对象（down vs next）、次调 dt（down−σ vs next−σ）、噪声注入（有 vs 无）、interpol 表头（无 [0,…] 死头 vs 有）、辅助表数量（三张 vs 一张）；首调中点读格（sigIE[i] vs sigIE[i+1]）。
- **522 `diff-euler-ancestral-step`**：单调用 Euler ancestral，噪声拆分在 σ→(σ_down, σ_up) 上同构，但无中点/无两调用状态机；DDPM ancestral 整段循环已判不立项。
- **643 `diff-heun-predictor-corrector`**：导数平均 vs 丢弃、算术中点 vs σ_down 插值。
- agentic/kvcache 无重合。

## 测试与变异

14 用例：oracle 全对拍（numpy float64 复刻三表构造 + 两步状态机）、首调半步/暂存样本/忽略 noise、次调 down 步长/噪声注入/从暂存出发（`prev − noise·up − deriv·dt` 反解回原样本）、σ_down = σ_next²/σ 与方差守恒、末逻辑步落 x0、v_prediction 双调用、noise 四种非法、prediction_type 校验、奇偶/范围/越界（含 i 恰为 2N、消息断言「末逻辑步」防范围松动变异空抓）、state 四种非法、σ 表六种非法、三步全循环（2N−1 次调用）、端到端。

33 条变异 **30 CAUGHT + 3 等价**：

| 等价变异 | 实证 |
|---|---|
| `m-down-tail`（删 `down[-1]=0`） | 300 组随机：interpol 逐元素不变（nan 被 `[-2:]` 清吸收）；down[N]=nan 只落展开表 2N−1..2N+1 格，次调最大读 `sig_de[2N−4]`，全部不被读 |
| `m-first-dt-full`（首调 dt 用 `sig_e[i+1]`） | interp[j] 与 σ_{j+1} 的偏差经 dt 相对化后 = O(1e−16/(σ_next/σ))，任意合法输入 ≤ 1e−15；这正是「interpol ≡ σ_next」恒等式的体现（float32 域 4e-6 的偏差被 `_close` 相对容差吸收） |
| `m-sigma-interpol-guard`（`<= 0` → `< 0`） | 合法次调 j ≤ N−2 时 interp[j] = σ_next[j] > 0（校验强制前 N 个 σ 为正），300 组 min = 0.016；interpol = 0 的格只被首调（不做正性检查）读 |

## 踩坑

- **interpol 表头索引**：ancestral 的 interpol 没有 754 那个手动 `[0,…]` 死头——首调读 `sigIE[i]`（不是 `i+1`），首调与次调读同一格。首版照抄 754 的 `i+1`，5 个用例齐刷刷 FAIL，回源重读 L621 才发现。
- **oracle 展开表索引**：oracle 次调曾用未展开基础表的 `up[i−1]`——j≥1 时错位（j=0 恰好同值导致部分用例假通过）。展开后 `sig_ue[2j] = up[j]`。
- **torch.lerp 的 −inf 行为**：实测 `lerp(有限, −inf, 0.5) = nan`——与 numpy 公式 `a + w(b−a)`（给 −inf）和凸组合 `a(1−w)+bw`（给 −inf）都不同。这决定了 `[-2:]` 清场是两格刚需。
- reject 用例防空抓：`m-second-range` 松动后 i=2N−1 滑到「中点为正」检查仍抛 ValueError → 消息断言改「末逻辑步」；「首调索引恰为 2N」用例防 `m-first-range` 松动（σ=0 走到除零而非抛错）。
