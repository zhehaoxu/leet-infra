# notes — ocr-mtmd-image-batching（order 690）

## 回源

- 出处：`ggml-org/llama.cpp` `tools/mtmd/mtmd-helper.cpp` 的 `mtmd_helper_eval_chunks → eval_chunk_single → decode_image_chunk`（GitHub API contents + base64）；批结构 `decode_embd_batch`/`set_position_normal(pos_0 + i)` 在 `mtmd-helper-common.h`。
- 三段机制（逐行核对源码）：
  1. `eval_chunks`：空序列 return 0（"no chunks to eval"）；`chunk_logits_last = (i == n_chunks-1) && logits_last`；
  2. **text 分支**：`while (i < n_tokens)` 内圈 `n_added < n_batch` 填批，每 token `pos = n_past++` 逐位推进，`*new_n_past += n_added`；`logits_last && is_last_token` 时在**批内最后一个 add 的 token**（idx）设 output logits；
  3. **image 分支**：先 `mtmd_encode_chunk` 出整段 embedding，再 `decode_image_chunk`：`n_img_batches = (n_tokens + n_batch - 1) / n_batch`、批循环 `pos_offset = i_batch*n_batch`、`n_tokens_batch = min(n_batch, n_tokens - pos_offset)`；位置由 `set_position_normal(n_past, ...)` 一次设 `pos_0 + i`（批只取视图偏移），**批完后 `n_past += n_pos`（非 M-RoPE = n_tokens）**。
- **官方 quirk（已钉题）**：image/audio 分支不使用 `logits_last` 参数——最后 chunk 是 image 时无 logits 输出；空 image chunk（n_tokens=0）批数为 0 且 n_pos=0，整 chunk 空转。
- M-RoPE 分支（`n_pos_per_embd = 4`、`set_position_mrope_2d` 的 t/y/x/z 四维）不回源出题——归 omni 系（`omni-tmrope-position` 已占）；本题钉非 M-RoPE 路径并在题面注明。
- 与 `inf-*` 调度系划界：continuous batching 是跨请求调度，本题是**单请求内 multimodal chunk 的批切分与位置账**；与 `mm-mrope-video-text-deltas` 划界：那边考 position_ids delta 账，这边考**批调度与 n_past 推进**。

## 判题面

- solve 入参：chunks（[{"type": "text"|"image", "n_tokens": int}, ...]）、n_batch（正 int，官方 GGML_ASSERT(n_batch > 0) 转题面 ValueError）、logits_last（默认 False）、n_past（默认 0）。返回 batches/new_n_past/logits_pos。
- oracle：`test_cases._py` 用事件表路径独立重算——先按 chunk 枚举全部批切分边界（text 贪心满批、image ceil + 末批余数），pos 账最后从事件表统一累计；300 组随机对拍。
- 锚点：[text 5, image 576, text 3]、n_batch=256 → 批 5 / 256(5)/256(261)/64(517) / 3(581)，logits_pos=583，new_n_past=584；image-only + logits_last=True → logits_pos=None（quirk）；恰整除 512/256 两满批；余 1 的 513/256 末批单 token；text 7/n_batch=3 → 3/3/1 且 logits 在批 3；n_past=100 平移；空序列/空 text chunk/空 image chunk 三种空转；n_batch=1 逐 token 批。
- 非法输入：chunks 非列表/type 非法/n_tokens 负或非 int/n_batch ≤ 0 或 bool/logits_last 非 bool/n_past 负 → ValueError。

## 变异记录（12 条全抓）

ceil-to-floor（批数 floor 丢末批）、last-batch-overshoot（末批越界）、image-pos-no-offset（批 pos 不加视图偏移）、image-pos-no-advance（image 后位置不推进）、text-logits-every-batch（漏 is_last_token）、logits-ignore-last-chunk（漏 chunk_is_last）、text-batch-off-by-one（≤ 越界）、image-npos-by-batches（推进量换批数）、text-pos-start-at-end（text 批 pos 起点错）、kind-mislabeled、logits-pos-first-token（批首替批末）、mrope-npos-times4（n_pos 误 ×4）。

## 红线

embedding/token id 数值、mmproj 编码、M-RoPE 四维位置不出题；音频走 image 同路径不单列（题面注明）；continuous batching/跨请求调度归 inference 系；`omni-tmrope-position` 已占 M-RoPE 位置账、`mm-mrope-video-text-deltas` 已占 delta 账勿混。
