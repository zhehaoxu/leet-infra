"""order 843 骨架：Captum LRP 的 z-rule relevance 传播账。

你要实现 Captum 官方 `captum/attr/_core/lrp.py` 里 LRP 的 relevance 传播账，
以及 `captum/attr/_utils/lrp_rules.py` 里三条规则的权重改写。

四个 mode（都从 solve 入口进）：

  rewrite_rule    : 规则权重改写账
      solve("rewrite_rule", rule=..., weights=..., bias=..., gamma=...,
            set_bias_to_zero=...)
      -> {"weights": [[...]], "bias": [...]}

  z_rule_layer    : 单层线性层的 z-rule relevance 分解
      solve("z_rule_layer", weights=..., bias=..., inputs=..., target=...)
      -> {"outputs": [...], "relevance_out": [...], "relevance_in": [...],
          "attribution": [...]}

  stack_relevance : 多层线性栈的 z-rule 传播账
      solve("stack_relevance", layers=[{...}], inputs=..., target=...)
      -> {"activations": [[...]], "outputs": [[...]],
          "relevance_out": [[...]], "relevance_in": [[...]],
          "attribution": [...]}

  completeness    : 完备性 delta
      solve("completeness", attributions=[...], output=...)
      -> {"attr_sum": ..., "delta": ...}

口径要点（详见 description.md）：
  - 所有输入量都是整数；eps 取精确有理数 1/10^9。
  - 全链路建议用 fractions.Fraction 精确计算，最后统一转 float。
  - 官方 LRP 走**两次前向**：第一次原权重前向存激活，第二次「改写权重 × 原激活」。
  - sign(0) 强制为 +1；delta = output - sum(attributions)（output 在前）。
"""

from fractions import Fraction

EPS = Fraction(1, 10 ** 9)
SUPPORTED_RULES = ("epsilon", "gamma", "alpha1_beta0")


def solve(mode, **kw):
    raise NotImplementedError("请实现 solve(mode, **kw)")
