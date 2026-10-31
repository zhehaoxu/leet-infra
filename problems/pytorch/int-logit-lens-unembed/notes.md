# notes — int-logit-lens-unembed（order 661）

## 回源

- 出处：nostalgebraist 的 LessWrong post「interpreting GPT: the logit lens」（2020）；transformers 无独立官方仓，本题按「final norm + unembed 显式入参」的简化口径出题（todos #103 红线原文）。
- 与 653 `int-tuned-lens-affine` 划界：tuned lens 是**学出来的仿射探针**（per-layer W_l/b_l，无 final LN），本题是**免训练读出**（复用 final LN），账本侧考 agree/rank 而非 CE/KL。

## 判题面

- solve 入参：hiddens `[L,D]`、lnf_w/lnf_b `[D]`、unembed `[D,V]`、apply_norm bool、eps>0（缺省 1e-5）。返回 logits/argmax/final_argmax/agree/rank_of_final/top_logit（argmax 类为精确比较，不做容差）。
- oracle：`test_cases._py` 纯 Python（math.sqrt + 手写 argmax），对拍全部字段。
- 锚点：① h=[4,0]：normed=±2/√(4+1e-5)、logits=[e,−e,0]、两层 argmax 相异（agree=[False,True]）、rank_of_final=[3,1]；② apply_norm=False 的 raw 读出对照；③ 全并列 logits → argmax=0（最小下标钉死）；④ rank==1 ⇔ agree（自洽不变量）；⑤ b 直接平移线性组合；⑥ 常数隐态 var=0 只由 eps 与 b 决定。
- 非法输入：形状、空输入、apply_norm 非 bool、eps≤0 → ValueError。

## 变异记录（9 条全抓）

no-layernorm（logit lens 最经典复现错误：跳过 final LN）、var-unbiased（除 D−1）、eps-drop、no-affine（丢 w/b）、tie-largest（并列取大）、rank-inclusive（≥ 计数含自身）、agree-inverted、final-from-first（参照层取第 0 层）、unembed-transpose。

## 边界与陷阱

- 锚点手算必须带 eps：±2/√(4+1e-5) ≈ ±0.99999875，1e-12 级断言用解析式。
- argmax 并列行为 torch 无保证，题面钉死「最小下标」并手写实现。
- 红线：真实模型 forward、tuned lens 的探针参数训练、软标签 CE/KL（已归 653）不出。
