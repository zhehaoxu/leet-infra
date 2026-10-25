# 解题说明 · Random Eviction

## 为什么"最简单的基线"必须存在

评测 KV 压缩论文时，最阴险的假阳性是：打分器与"最后 query 的注意力"高度相关，而后者本来就被 recent bias 支配——随便留最近的 token 就能拿高分。`RandomPress`（kvpress `random_press.py`）用纯随机分数 top-k 提供一个**无信息下界**；学界标准做法（LongBench/RULER 报告里常见）是再叠上 StreamingLLM 的头尾保护，因为连随机基线也几乎从不真正驱逐 sink/recent——它们的注意力占比大到让"保留它们"成为免费操作。本题就是这个合成形态。

## 随机性的工程处理

kvpress 用 `torch.Generator(seed)` 复现随机；刷题判题更进一步：**连 generator 都不许用**，均匀抽样结果 `draws` 直接作为输入张量给出。这样 (a) 参考解与用户解在完全相同的"骰子"上对拍，(b) 失败用例可精确重放，(c) 不存在 torch 版本间 `rand` 序列漂移问题。选"top-q 大 draws"就是把"均匀无放回抽样 q 个"固化为确定操作（Fisher–Yates 的等价排序视图）。

规则里最容易被忽略的分支：
- 名额 \(q=\max(0,n\_kept-|F|)\)：**先扣头尾再抽签**，而不是对全序列抽签。
- \(|F|>n\_kept\)：保留数**可以超预算**——StreamingLLM 保护优先（对比：`kv-chunk-independent-topk` 里每段保底 1 也是"保护优先于比例"的同族设计）。
- coverage 的 0/0 兜底：`attn_weights` 全零时用 \(|kept|/L\)，把"注意力覆盖"退化成"位置覆盖率"。

## 与近邻题对比

- `kv-streaming-llm-sink`（145）：给「sink+滑窗=1、中段=0」的**确定性打分**；本题把中段的 0 换成随机抽签，是 145 的随机化扩展。
- `kv-knorm-importance`（146）、`kv-observed-attention`（149）等：所有信息型 scorer 在论文消融里都必须与本题比较——**如果 attention coverage / 下游准确率与随机基线差距小于噪声，方法无效**。
- coverage 指标本身即是 SnapKV/PyramidKV 论文常用的"重建误差"代理：用最后一问的 attention 分布衡量被驱逐 token 携带的信息比例。

## 陷阱

- `draws` 平局（构造的全相等向量）：必须按 (−draw, position) 取左；纯 `torch.topk` 不保证。
- 区间重叠：\(n\_sink+window\_size\ge L\) 时中段为空、\(F\) 覆盖全集，`kept` 长度是 \(L\) 而不是 \(n\_sink+window\_size\)（去重！）。
- coverage 是标量 float，别返回 0 维张量比较失败——测试用 `abs(got-exp)<=1e-5`。
