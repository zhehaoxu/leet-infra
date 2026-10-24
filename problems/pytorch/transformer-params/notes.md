# 解题思路

## 面试官意图

「给你一个 Transformer 配置，说出它有多少参数」是基础设施/训练岗的高频白板题。它考察
你是否真的理解每个模块的权重矩阵形状，而不是只会背「GPT-2 是 1.2 亿」。本题把结构固定成
一个清晰的口径，要求你把 **attention、FFN、LayerNorm、嵌入层** 四类参数分别清点再汇总。

## 逐项拆解

一个 decoder block 里可学习的权重：

1. **Self-Attention**：$q,k,v$ 三个投影把 $h\to h$，输出投影 $o$ 也是 $h\to h$，
   共 $4$ 个 $h\times h$ 矩阵 → $4h^2$。注意这与头数无关：无论切几个头，
   `q_proj` 的形状始终是 `[hidden, hidden]`。
2. **FFN（MLP）**：先升维 $h\to f$ 再降回 $f\to h$ → $2hf$。
3. **LayerNorm ×2**：每个 LN 有 weight 和 bias，各 $h$ 个 → $2\times 2h = 4h$。

把三项相加就是 $P_{\text{layer}} = 4h^2 + 2hf + 4h$。乘上层数得到所有 block 的参数。

## 嵌入与输出头

- 输入 token 嵌入是 `vocab × hidden`。
- 输出 LM head 也是 `vocab × hidden`。**tied embedding**（权重共享）让两者用同一张表，
  所以只计一次；不共享时再补一份 `vocab·h`。GPT-2 是共享的。
- 网络最后还有一个 final LayerNorm：$2h$。

## 数量级直觉

对 GPT-2 small（$h=768, f=3072, L=12$）：块参数 $84.97$M 远大于嵌入 $38.6$M，而
$4h^2$ 与 $2hf$ 又各占块参数的大头——所以当 $f=4h$ 时，attention 的 $4h^2$ 和 FFN 的
$2h\cdot 4h = 8h^2$ 之比约为 $1:2$。这也是「FFN 才是参数量大户」的经验来源。

## 常见错误

- 忘记 LayerNorm / final LN：小模型上占比不大，但会被对拍抓出。
- 把 tied/untied 搞反：untied 一定比 tied 多整张 `vocab×h`。
- 错误地按头数乘参数：MHA 的投影恒为 $4h^2$，与 heads 无关。

## 判题口径

`test_cases.py` 用一个**逐矩阵累加**的独立参考 `_ref` 与公式实现交叉核对，确保两条推导路径
一致；`check_untied_adds_head` 单独验证共享/解绑的差值恰为一张嵌入表。
