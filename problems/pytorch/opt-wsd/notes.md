# 详解：WSD 预热-恒定-多项式退火

## 出处
- 三段式 WSD 的提出与 LLM 预训练用法：Hu et al., *MiniCPM: Unveiling the Potential of Large Language Models with Scalable Training Strategies*, arXiv:2404.06395（原文明确写出 "Warmup-Stable-Decay (WSD) learning rate scheduler, conducive to continuous training and domain adaptation"）。
- WSD 的系统分析（恒定支路 + 末端退火、与余弦的对比）：Wen et al., *Understanding Warmup-Stable-Decay Learning Rates*, arXiv:2410.05192。
- 参考实现：HuggingFace `transformers/optimization.py::_get_wsd_scheduler_lambda` / `get_wsd_schedule`（`warmup_type/decay_type ∈ {linear, cosine, 1-sqrt}`，`linear` 即 \((1-t)^1\)）；kozistr/pytorch_optimizer `pytorch_optimizer/lr_scheduler/wsd.py`（cooldown `linear`/`1-sqrt`/`1-square` 同一族）。
- 注意：todos.md 里给的 arXiv:2110.07621 **不是** WSD 论文（那是 Heller et al. 的流体力学文章），本题按上面三个可核验出处实现。

## 核心与易错点
1. **三段账**：`warmup_steps`、`stable_steps`、`decay_steps` 是三个**长度**，段边界是左闭右开：\(t\in[0,W)\) warmup、\([W,W+S)\) stable、\([W+S,W+S+D)\) decay、之后恒为地板。恒定段是 WSD 的灵魂：它可以无限延长，退火只在末端发生，所以能"一个恒定支路 + 多条退火支路"地做续训与超参搜索。
2. **退火写的是 \((1-t_{\text{decay}})^p\)，不是 \(1-t_{\text{decay}}^p\)**。这两个是不同族曲线：\(p=1\) 时二者重合，\(p=2\) 时前者是 \((1-\tau)^2\)（末端斜率 0、贴地更软），后者是 \(1-\tau^2\)（末端陡降）。实现里最容易写反。
3. **地板下移**：`min_lr_ratio` 不是 `min_lr`，而是倍率。warmup 段从 `base_lr·m` 起步（\(t=0\) 时不是 0），decay 段以 `base_lr·m` 收尾，公式为 \(f(1-m)+m\)；stable 段仍是纯 1。
4. **步数约定**：「构造即第 0 步」，`step()` 才前进——与 `opt-warmup-cosine` 同口径，别写成先取后加或先加后取的错位序列。
5. **除零**：`decay_steps=0` 时不能算 \(\tau\)，直接进地板分支；`warmup_steps=0` 时 \(t=0\) 就已在恒定段。

## 对拍口径
- 主口径：**独立纯 Python 重算** \(\text{base\_lr}\cdot f(t)\)（test_cases 的 `_oracle` 用「先算原始倍率、再统一夹紧」的不同结构实现），逐点绝对误差 < 1e-9。
- 硬参照：\(p=1\) 等价于 `transformers.get_wsd_schedule(warmup_type="linear", decay_type="linear")`，评测把它作为外部 oracle 逐步对拍（1e-12）。注意 HF 的 `LambdaLR` 也在构造时就写回 \(t=0\) 的 lr，所以序列索引天然对齐。
- 形态断言：warmup 段严格递增、恒定段完全相等、退火段单调不增、超出总步数后恒为 `base_lr·min_lr_ratio`。

## 复杂度
每步 \(O(1)\)（每个 param group 一次闭式计算），无额外状态；序列长度 \(W+S+D\) 的账全在索引上，不涉及张量。
