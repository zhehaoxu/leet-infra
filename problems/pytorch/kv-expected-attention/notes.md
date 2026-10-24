# 解题说明 · ExpectedAttention

## 期望注意力的推导

若 $q \sim \mathcal{N}(\mu, \Sigma)$，则对固定 $k$：$q^\top k \sim \mathcal{N}(\mu^\top k,\, k^\top \Sigma k)$。因此：

$$ \mathbb{E}[e^{q^\top k}] = e^{\mu^\top k + \frac{1}{2} k^\top \Sigma k} \quad\text{(高斯矩生成函数)} $$

除以 $\sqrt{d}$ 得到题目里的 $u_{h,n}$。

## RoPE 平均

真实 kvpress 会先对未来 $n_\text{future}$ 个位置的 RoPE 旋转矩阵求平均 $\bar{R}$、把 $\mu$ 与 $\Sigma$ 变换到"未来坐标系"（`mu ← mu @ R.T`、`cov ← R @ cov @ R.T`）。**题目为便于对拍，直接把变换后的 mu / cov 作为输入**——不实现 RoPE 部分。

## vnorm 缩放

原始 attention 分数只反映"多少注意力"，不反映"value 有多大"。用 $\|v\|_2$ 缩放让"高注意力 × 大 value"的 pair 排到最前。加 $\varepsilon$ 防止 $\|v\|=0$ 或 $\text{softmax}=0$ 时分数直接归零。

## Sink pad

前 `n_sink` 位（默认 4）不看内容一律给最高分——与 StreamingLLM 一样保护 attention sink。分数是 **全局 max + 1**，因此 topk(n_kept) 一定优先保留它们。

## kv-group

真实实现里 `num_key_value_groups = num_heads / num_kv_heads`，softmax 后要沿 group 平均。题目固定 `num_heads = num_kv_heads`（不做 group 折叠）。
