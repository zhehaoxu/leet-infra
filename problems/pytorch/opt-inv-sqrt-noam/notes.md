# 详解：Noam 逆平方根学习率

## 出处
- Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762，**式 (5)**：
  `lrate = d_model**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))`，
  并配文 "step_num starts at 1"。论文用的 \(d_{model}=512,\ warmup=4000\)，峰值约 0.0007。
  本题公式与之逐字一致，唯一调整是**步编号从 0 起**（沿用本合集"构造即第 0 步"的约定）：
  \(t\ge1\) 时与论文完全相同，\(t=0\) 由线性支给出 0（即论文的 step 1 之前那一点）。
- 同族复用该表达式的第二处：Shazeer & Stern, *AdaFactor*, arXiv:1804.04235 §4.1 的相对步长
  \(\min(s^{-1/2}, s/warmup^{3/2})\)（`opt-adafactor` 内嵌的就是它，但那是"更新幅度"而非独立调度器）。
- 外部数值参照：`transformers/optimization.py::_get_inverse_sqrt_schedule_lr_lambda`
  （由 big_vision 改写而来）：`timescale` 默认取 `num_warmup_steps or 10_000`，于是
  \(t<w\) 返回 \(t/w\)、\(t\ge w\) 返回 \(\sqrt{w/t}\)。把它乘上峰值 \(P=d^{-1/2}w^{-1/2}\) 就是 Noam：
  \(P\cdot t/w=d^{-1/2}t\,w^{-3/2}\)、\(P\cdot\sqrt{w/t}=d^{-1/2}t^{-1/2}\)，两支逐字对上。
  实测两者相对偏差 \(\sim2\times10^{-16}\)（纯浮点落点差），故可作逐步对拍参照。
- todos.md #90⑤ 给的表达式 \(d^{-0.5}\min(t^{-0.5},t\,w^{-1.5})\) 正确，无需更正；缺口只在
  "它不是独立调度器"——合集里此前没有 Noam 型调度器，本题补上。

## 核心与易错点
1. **\(t=0\) 不能真的代入 min 式**：`0**-0.5` 在 Python 里直接 `ZeroDivisionError`。必须按分段式写，
   线性支给出 \(lr(0)=0\)；这是本题最常见的翻车点。
2. **warmup 指数是 \(-1.5\) 而不是 \(-0.5\)**：\(t\cdot w^{-3/2}\) 在 \(t=w\) 时等于 \(w^{-1/2}\)，
   正好与衰减支相接。写成 \(w^{-0.5}\) 会让 warmup 段整体抬高 \(w\) 倍、峰值处出现跳变，
   评测用"相邻差在 \(t=w\) 仍等于 warmup 公差"这条卡住它。
3. **峰值位置**：两支相交于 \(t=w\)，所以 `argmax(lr)==warmup_steps`；升段是**等差**（线性）序列，
   不是平方根升段——把 \(t^{-1/2}\) 反过来用当 warmup 是常见错误。
4. **绝对学习率，不乘 base_lr**：与 WSD/OneCycle 那些"倍率型"调度不同，Noam 的 \(lr(t)\) 由 \(d,w\) 定死，
   与优化器里原来的 lr 无关（多参数组也写同一个值）。
5. **无总步数、无夹紧**：\(t\to\infty\) 只是继续 \(t^{-1/2}\) 下降，永远为正；不要加 `min_lr`、
   也不要在超过某步后钉死。长跑不变量 `lr(t)*sqrt(t) == d**-0.5` 专治此类自作主张。
6. **\(w=1\) 是合法退化**：第 1 步即达峰、之后纯衰减；`model_dim` 与 `warmup_steps` 都必须是正整数。

## 对拍口径
- 独立重算（主参照）：`_oracle` 直接按论文 min 闭式 \(d^{-1/2}\min(t^{-1/2},t\,w^{-3/2})\) 计算（\(t=0\) 特判 0），
  与解答的分段实现结构不同；容差 \(10^{-12}\)（对极小量用相对尺度），覆盖 6 组 \((d,w)\) 含 4000 步 warmup、\(w=1\)。
- 外部参照：`transformers.get_inverse_sqrt_schedule`，优化器 `SGD(lr=d**-0.5*w**-0.5)`、`timescale=warmup_steps`，
  按同一约定读 40 步，`torch.allclose(rtol=1e-12, atol=1e-15)`，4 组配置。
- 示例数值由参考解实跑得到（`model_dim=64, warmup_steps=5`，8 位小数写进题面）。
- 退化/不变量：\(lr(0)=0\) 且不崩；warmup 段等差（公差 \(d^{-1/2}w^{-3/2}\)）；峰值恰在 \(t=w\) 且该点前后差连续；
  衰减段严格下降恒正；跑到 \(t=604\) 仍有 `lr*sqrt(t)==d**-0.5`、未被夹紧。

## 复杂度
每步 \(O(\text{param groups})\)，常数次浮点幂运算，无状态张量、无历史依赖（闭式只由 \(t\) 决定），
因此天然支持从任意步续训；\(T\) 步总代价 \(O(T)\)。
