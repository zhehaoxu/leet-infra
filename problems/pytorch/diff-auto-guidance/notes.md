# 题解笔记：AutoGuidance 的锚点与恒等阈值一起翻转

## 出处

- HuggingFace diffusers v0.42.1 `src/diffusers/guiders/auto_guidance.py`：类文档第 34 行给出论文链接，`_input_predictions` 在第 64 行，`forward` 在 156–170，`num_conditions` 在 176–181，`_is_ag_enabled` 在 182–198。
- 同库 `src/diffusers/guiders/guider_utils.py`：`BaseGuidance.__init__` 的 `start / stop` 校验在 59–63，`set_state` 在 100–104，`rescale_noise_cfg` 在 374–396。
- 论文 Karras 等《Guiding a Diffusion Model with a Bad Version of Itself》，arXiv:2406.02507（已核对标题与作者）。标准差回缩出自《Common Diffusion Noise Schedules and Sample Steps are Flawed》，arXiv:2305.08891（同样已核对）。

## 核心与易错点

- **一次翻转改两件事**。`use_original_formulation` 既把外推锚点从 `pred_uncond` 换到 `pred_cond`，也把恒等尺度从 1.0 换到 0.0。二者必须成对出现：`isclose(s, 0.0 if original else 1.0)`。只翻锚点不翻阈值，会让 `s=0` 的「关掉引导」变成「输出乘上一份 shift」；只翻阈值不翻锚点，`s=1` 会被当成关闭而实际上差着一个 `pred_cond - pred_uncond`。
- **`math.isclose` 的两种不对称**。默认 `rel_tol=1e-9`、`abs_tol=0`：拿它跟 1.0 比是相对判定，`1+1e-10` 判成恒等、`1+1e-8` 判成非恒等；拿它跟 0.0 比时 `rel_tol * |b| = 0`，于是只有**恰好** 0.0 才算恒等，`1e-12` 仍然开着。`check_isclose_is_relative` 把这两条都压了一遍，其中原表述 `s=1e-12` 那条只能按绝对容差比（结果与 `pred_cond` 差 5e-13 量级）。
- **恒等分支照样跑 rescale**。上游把 `if guidance_rescale > 0.0` 写在 `if/else` 之外，所以关闭时也会把 `pred_cond` 送进 `rescale_noise_cfg`。数学上 `std(pred_cond)/std(pred_cond) = 1`，是恒等；浮点上 `r*x + (1-r)*x` 在 `r=0.5` 这类二进制友好系数下逐位相同，在 `r=0.3` 下可能差 1 ulp。`check_rescale_runs_on_off_branch` 分别用逐位相等与 `atol=1e-7` 各卡一次。
- **窗口是 `int` 截断的左闭右开区间**。`int(0.1*20)=2`、`int(0.8*20)=16` → 生效步是 `[2,16)`。`start` 与 `stop` 都在构造期校验（`start` 严格小于 1、`stop` 不小于 `start`），而 `step` 只在运行期参与比较，两套校验来自两个不同的函数。
- **弱预测这一支要不要真的跑模型，取决于 `num_conditions`**：上游在第 176–181 行按开关决定前向次数，关闭时 `forward` 只收到 `pred_cond`，`pred_uncond` 连形状都不会被检查——这正是题面允许关闭时传 `None`、甚至传错形状的依据。

## 与上游的差异

- 上游的「坏模型」是靠 `_apply_layer_skip_hook` 在若干 block 上按 `dropout` 概率丢层跑出来的；本题把这份弱预测直接作为 `pred_uncond` 入参，去掉了 hook、`LayerSkipConfig` 与 `prepare_models / cleanup_models` 这一整套装配。
- 去掉了 `BaseGuidance` 的 `enabled` 开关（它只是与 `within / close` 相与的第三个布尔），也去掉了 `GuiderOutput` 三元组，只留最终预测。
- 数值上强制 float64 中间量并返回 float32；上游按传入 dtype 计算。同时要求 batch 之外元素数 ≥ 2：一维输入或 `(B,1)` 输入在 `rescale_noise_cfg` 里会因自由度 ≤ 0 直接得到 NaN，本题改成 `ValueError` 提前拦住。
- `guidance_rescale` 在源码里没有上界检查，本题按同族题目（`diff-guidance-rescale`）的惯例收窄到 `[0, 1]`；`start / stop` 保持与 `BaseGuidance.__init__` 一致，即 `start` 不能取 1.0。

## 与其他题的分工

- `diff-cfg` 只练 `eps_u + w·(eps_c - eps_u)` 这一条公式；本题的全部难点在「锚点与阈值成对翻转 + `isclose` 的相对/绝对不对称」，同款公式只是外壳。
- `diff-guidance-rescale` 负责标准差回缩本身（连续 `t` 闭区间、`dims` 可配、φ 混合），本题只把它当作开关尾部的一个可选步骤，且 `dims` 固定为除 batch 外的全部维度。
- `diff-pag-dual-guidance` / `diff-ltx2-guidance` 处理多分量合成与各自窗口，本题是单分量、双表述的对照版本；`diff-mambo-guidance` 则研究尺度随幅值比自适应，与这里的固定尺度不同。
