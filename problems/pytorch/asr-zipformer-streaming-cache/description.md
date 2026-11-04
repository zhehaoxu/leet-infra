# Zipformer 流式缓存状态账

## 背景

Zipformer2（k2-fsa/icefall，`egs/librispeech/ASR/zipformer/zipformer.py`）的流式推理不是传统 KV cache，而是**固定长度的滚动左上下文**：每层缓存 6 组张量（cached_key、cached_nonlin_attn、cached_val1、cached_val2、cached_conv1、cached_conv2），每 chunk 拼接新帧后取尾部 L 帧作为新缓存。本题重建其中可判定的账目：缓存滚动、相对位置对齐、门控卷积、chunk 级因果卷积分解与整除恒等。

## 机制拆解

### 1. 滚动缓存（cache_roll / attn_scores / value_out / nonlin_out）

官方所有缓存更新共享同一句法（以 key 为例）：

```python
k = torch.cat([cached_key, k], dim=0)     # (L+seq, b, dim)
cached_key = k[-left_context_len:, ...]   # 尾部 L 帧
```

- 缓存长度恒为 `left_context_len`（conv 缓存是 `left_pad = kernel_size // 2`）；
- **不是 KV cache**：旧帧被挤出，模型对 L 帧之前的内容不可见（流式精度损失来源）；
- `left_context_len = left_context_frames // downsampling_factor`（encoder 侧按 stack 降采样折算）。

### 2. 相对位置对齐（attn_scores）

`RelPositionMultiheadAttentionWeights.streaming_forward` 中 pos_scores 的最后一轴是**相对位置**，官方用 `as_strided` 把它折叠成绝对 key 索引：

```python
pos_scores = pos_scores.as_strided(
    (h, b, seq, k_len),
    (s0, s1, s2 - s3, s3),
    storage_offset=s3 * (seq_len - 1))
```

等价的显式 gather：`indexes[t, j] = (seq-1-t) + j`。推导：连续张量按官方 stride 展开后，`[h,b,t,j]` 处元素来自原张量 `[h, b, t, (seq-1-t)+j]`。相对位置的取值域 `[(seq-1)-t, L+2seq-2-t] ⊂ [0, 2seq-2+L]` 恰好落在 pos_emb 长度内。打分 = `q@k + pos_scores`，padding mask 填 `-1000` 后 softmax。

### 3. 门控卷积（conv_gated）

`ConvolutionModule.streaming_forward` 的门控段：in_proj 出 `2*channels`，后半走 sigmoid 作逐通道门 `x = x * sigmoid(s)`；padding 帧在进 depthwise conv 前 `masked_fill(..., 0.0)`（否则 padding 会泄进卷积的加权和）。

### 4. chunk 级因果卷积（chunk_causal_conv / chunk_scale / stream_equiv）

`ChunkCausalDepthwiseConv1d` 把一个 k 阶 depthwise 卷积拆成两条支路相加：

- **causal 支路**：half-kernel（`(k+1)//2` 阶）真因果卷积，左侧补 `left_pad = k//2`（流式时该 pad 由 cache 提供）；
- **chunkwise 支路**：对称 k 阶卷积（构造时自带 `padding=k//2`），只在块内看邻居，输出乘 `chunk_scale`。

`_get_chunk_scale(scale, cs)` 返回 `(channels, cs)`：

```
1.0 + left_edge(头修正, 长 k, 右侧补零到 cs) + right_edge(尾修正, 长 k, 左侧补零到 cs)
cs < k 时截断：left 取前 cs、right 取后 cs
```

cs 与 k 的重叠区两修正相加（cs=8, k=5 时列 3-4 重叠）。

**整除恒等**：`seq % chunk_size == 0` 时，逐块流式（cache 链）拼接与官方一次性 `forward(x, chunk_size)` 逐步相等——这是流式设计正确性的核心，也是本题判题的最强断言。非整除时官方 forward 用 `right_pad = -seq % chunk_size` 补齐分块（流式无此步）。

## 判题域

- 所有权重（causal_w/chunk_w/scale/in_proj_w 等）与输入张量显式入参，fp64；
- 断言全部为闭式对拍 / 恒等式，无 RNG 漂移锚值；
- 7 mode：cache_roll、attn_scores、value_out、nonlin_out、conv_gated、chunk_scale、chunk_causal_conv、stream_equiv；
- 不出：真模型前向、subsample 前端账、非整除 right_pad 的流式行为、训练期 Balancer/Whiten。
