# 解题思路

## 面试官意图

「128K 上下文的推理为什么这么吃显存？」核心就是 KV cache。它随**序列长度线性增长**，长上下文
或大 batch 时会反超模型权重。本题还串起 MQA/GQA——它们正是为压 KV cache 而生的注意力变体
（与 `mqa` 那道题互为表里：一个讲计算，一个讲显存）。

## 尺寸怎么拼

缓存里对**每一层、每一个 KV 头、每一个已生成 token**都存了一条 K 向量和一条 V 向量，各
`head_dim` 维、每维 `dtype_bytes` 字节。于是

$$ \text{bytes} = 2_{(K,V)} \cdot L \cdot \text{kv\_heads} \cdot d_{\text{head}} \cdot \text{seq} \cdot \text{batch} \cdot \text{dtype\_bytes} $$

只要盯住「$K$ 和 $V$ 是两个」这个最常漏的因子 $2$，其余都是维度连乘。

## MQA / GQA 为什么省

MHA 里每个查询头都有独立的 KV，`kv_heads = heads`。

- **MQA**：所有查询头共用**一组** KV（`kv_heads=1`），缓存直接缩到 $1/\text{heads}$。例子里
  32 头 → 从 1 GiB 降到 32 MiB，32 倍。
- **GQA**：折中，把 `heads` 个查询头分成 `kv_heads` 组共享（如 Llama-3 8B 用 8 组），缩到
  $\text{heads}/\text{kv\_heads}$ 倍，质量损失远小于 MQA。

这正是测试 `check_mqa_gqa_ratios` 想固化的比例关系。

## 数量级直觉

对 7B 级（$L{=}32, n_h{=}32, d_h{=}128$, fp16）每 token 每份约
$2\cdot32\cdot32\cdot128\cdot2 = 524$KB… 乘以 seq=2048 就是 1 GiB（batch=1）。可见
**seq 翻倍 → KV cache 翻倍**，而权重不变——这就是长上下文 batching 的显存瓶颈，也是 PagedAttention
（vLLM）要动态管理 KV 显存的原因。

## 常见错误

- 漏掉 $2$（只算了 K 或只算 V）。
- 把 `head_dim` 当成 `hidden`；当题目直接给 `head_dim` 时不要再去除以 heads。
- MQA 时仍用 `heads` 而非 `kv_heads=1`。

## 判题口径

`test_cases.py` 用**逐层逐头累加**的独立参考核对闭式，并单测 `seq/batch` 的线性、`kv_heads=0`
边界与 MHA/MQA/GQA 的缩小比例。
