# llama.cpp mtmd 图像 token 分批调度账

## 题意

llama.cpp 的多模态库 mtmd 把多模态输入拆成 chunk 序列（文本 token 段 / 图像或音频占位段），再逐 chunk 按上下文批上限 `n_batch` 切批送入解码。图像 chunk 的特殊之处：先由 mmproj 编码出整段 embedding，再按批切片解码，位置只推进一次。本题转写 `mtmd_helper_eval_chunks / eval_chunk_single / decode_image_chunk` 的调度账：给定 chunk 序列、批上限与 logits 开关，输出批序列、位置推进与 logits 落点。

## 求解

口径题面钉死（与官方一致）：

1. **空 chunk 序列**：直接返回，无批、位置不动（官方 "no chunks to eval"）；
2. **text chunk**：while 循环按 `n_batch` 切批（每批 ≤ n_batch），位置**逐 token 推进**——批 k 的 pos 区间 `[n_past+已发, +批大小)`；logits 只在「本 chunk 是最后一个 chunk 且 `logits_last` 且本批是 chunk 内最后一批」时设在批内最后一个 token 上；
3. **image chunk**：`n_img_batches = ceil(n_tokens / n_batch)`，批 k 的 pos 区间 `[n_past + k*n_batch, +min(n_batch, 剩余))`——官方 `set_position_normal(pos_0 + i)` 对整 chunk 一次设位、批循环按视图偏移取用；**chunk 解码完后位置一次推进 `n_tokens`**（非 M-RoPE 下 `n_pos == n_tokens`）；**image 分支不处理 `logits_last`**——最后一个 chunk 是 image 时无 logits（官方 quirk：logits 通常靠图像后的文本尾）；
4. `logits_pos` = 最后一次标记 logits 的位置（未标记则 None）。

token id 与 embedding 数值不入题，只考调度与位置账。

## 数据规模

- 0 ≤ chunk 数 ≤ 10^3，单 chunk 0 ≤ n_tokens ≤ 10^6，1 ≤ n_batch ≤ 10^6。
- chunks 非列表、chunk 非字典、type 不在 {text, image}、n_tokens 非负 int、n_batch 非 ≤0 的 int、logits_last 非 bool、n_past 非负 int → `ValueError`。
