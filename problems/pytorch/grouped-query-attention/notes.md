# GQA / MQA 详解

## 三种注意力共享同一组 KV 头

- **MHA**：$n_\text{head}$ 个 query 头 ↔ $n_\text{head}$ 组 KV，KV cache 最大。
- **MQA**：$n_\text{head}$ 个 query 头 ↔ **1** 组 KV，KV cache 最小但质量下降明显。
- **GQA**（Mistral / Llama-2-70B / Gemma）：$n_\text{head}$ 个 query 头 ↔ $g$ 组 KV，$1 < g < n_\text{head}$；每 $n_\text{head}/g$ 个**连续** query 头共享同一组 KV。

$$ h_i^{kv} = \left\lfloor \frac{i}{n_\text{head}/g} \right\rfloor,\quad i \in [0, n_\text{head}) $$

## 关键坑：`repeat_interleave` vs `repeat`

把 KV 沿 head 维扩到 query head 数时，PyTorch 语义：

- `kv.repeat_interleave(rep, dim=1)` → 每个 kv 头**连续复制 rep 次**（头 `[0,1]` 属 kv0、`[2,3]` 属 kv1）。
- `kv.repeat(1, rep, 1, 1)` 沿 head 维 → 循环拼接（`[0]` 属 kv0、`[1]` 属 kv1、`[2]` 又回到 kv0）。

GQA 用**前者**；判题里的分组归属测试专门扰动两组 kv 数值来检出把「连续分块」写成「循环取模」的错解。

## 与 MHA / SDPA 的退化关系

- `num_kv_heads == num_heads` → 应与标准多头注意力（每头独立 KV）一致，可用来回归。
- `num_kv_heads == 1` → MQA，最省 KV cache（推理长上下文时显存主要瓶颈）。
- 缩放必须是 $1/\sqrt{d_h}$，$d_h = \text{末维}/n_\text{head}$（不是末维本身、也不是 KV 末维）。

## 应用

现代 LLM 推理服务几乎都用 GQA：vLLM / TensorRT-LLM 的 PagedAttention 缓存按 **kv 头数** 分配，$g$ 越小 cache 越省，能塞更长上下文或更大 batch；从 MHA 权重转 GQA 一般通过「KV 头合并 / 折叠」得到（Ainslie 2023 的 upcasting 或 mean-pool）。
