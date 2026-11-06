# IndexTTS-2 的文本分段、静音收缩与 mel 长度账

## 背景

`index-tts/index-tts` 的 IndexTTS-2 在推理前后有三段与声学模型本身无关的纯序列/整数处理：文本 token 按标点优先级递归分段、生成 codes 里过长静音 token 的收缩、以及生成后按 `code_lens × 1.72` 换算 mel 帧数。本题把这三段抽成**无状态判题**：token 一律是字符串 id 列表，`codes` 一律是整数列表的列表，所有阈值 / 常量 / 停止符作显式入参；不跑真实 tokenizer、不加载声学模型、不读写音频。实现单入口 `solve(mode, **kw)`，共 3 个 mode。

「毫秒级时长控制」涉及训练式目标 token 预测器与 latent 插值，确定性差，不在本题范围。

## 通用约定

所有 mode 返回 dict，字段语义见「模式与规则」；未列出的中间量不判。逐 token 的 `for` 循环用 Python 语义（在 `for i in range(...)` 内写 `i += 1` **不改变下一轮迭代变量**，官方源码如此，必须如实复现）。常量默认值如下（判题也会显式传入，此处仅作口径说明）：

```
PUNCTUATION_MARKS_TOKENS = [".", "!", "?", "▁.", "▁?", "▁..."]
DEFAULT_MAX_SEG = 120
DEFAULT_RATIO = 1.72
DEFAULT_SILENT_TOKEN = 52
DEFAULT_MAX_CONSECUTIVE = 30
DEFAULT_MAX_MEL_TOKENS = 1500
```

## 模式与规则

### 1. split_segments：按标点优先级递归分段

输入 `tokenized, max_text_tokens_per_segment=120, split_tokens=None, quick_streaming_tokens=0`。`split_tokens` 为 `None` 时取 `PUNCTUATION_MARKS_TOKENS` 的副本。返回 `{"segments", "pre_merge", "total_token", "n_segments", "merge_flags"}`，其中 `pre_merge` 是合并前的段列表，`merge_flags` 与每段一一对应。

逐 token 累积 `cur` / `cur_len`，每轮按以下优先级判定（前两条的递归调用把 `cur` 用新的 `split_tokens` 重跑同一套逻辑；递归里 `split_tokens` 已含 `,` 或 `-`，故这两条不会再触发）：

```
not ("," in split_tokens or "▁," in split_tokens) and ("," in cur or "▁," in cur)
    -> sub = _split(cur, [",", "▁,"], ...)[0]
"-" not in split_tokens and "-" in cur
    -> sub = _split(cur, ["-"], ...)[0]
cur_len <= max_seg:
    token in split_tokens and cur_len > 2 -> split segment
    else continue
cur_len > max_seg -> hard-cut into max_seg blocks
```

- **标点切分**：命中 `token in split_tokens and cur_len > 2` 时，若 `i < len(tokenized) - 1` 且 `tokenized[i + 1] in ["'", "▁'"]`，先把该引号 token 追加进 `cur` 再执行 `i += 1`；随后 `segments.append(cur)` 并清空 `cur`。由于 `i += 1` 对 `for` 无效，下一轮仍会处理那个引号 token，须如实复现。
- **超长硬切**：`for j in range(0, len(cur), max_seg)`，若 `j + max_seg < len(cur)` 取 `cur[j:j + max_seg]`，否则取 `cur[j:]`；这些块一起 `extend` 进 `segments` 并清空 `cur`。
- 循环结束若 `cur` 非空（断言 `cur_len <= max_seg`）→ `segments.append(cur)`。

**后处理合并**：`total_token = 0`，遍历 `segments`：

```
total_token += len(seg)
if len(seg) == 0: continue
if len(merged) == 0:                      append;  flags "first"
elif len(merged[-1]) + len(seg) <= max_seg and total_token > quick:
                                          merge;   flags "merge_quick"
elif len(merged[-1]) + len(seg) <= max_seg / 2:
                                          merge;   flags "merge_half"
else:                                     append;  flags "new"
```

`max_seg / 2` 是**浮点除法**；`total_token` 是跨段累计值（不是当前段长度）；`quick_streaming_tokens=0` 时第一个合并条件几乎恒真，只有超长或空 `quick` 场景才体现第二个条件的阈值。

### 2. remove_long_silence：静音收缩与 batch 重拼

输入 `codes, silent_token=52, stop_mel_token, max_consecutive=30`。返回 `{"codes", "code_lens", "isfix", "counts", "kept_indices", "rebuilt", "max_len"}`：`counts` 是每行整行 silent 计数，`kept_indices` 是每行保留下来的下标列表，`rebuilt` 表示是否发生了 batch 级重拼。

```
len_ = code.index(stop_mel_token) if stop in code else len(code)
count = sum(1 for v in code if v == silent_token)
if count > max_consecutive:
    for k in range(len_):
        if code[k] != silent_token:      keep k; n = 0
        elif n < 10:                    keep k; n += 1
        else:                           drop
    len_ = len(kept_indices)
else:
    kept_indices = list(range(len_))
```

- `count` 是**整行全局计数**而非连续段长度，`stop_mel_token` 之后的部分不参与收缩但参与 `count`；因此「多段短静音」也会触发 `count > max_consecutive`。
- `n` 遇非静音归零（不是全局静音计数）；连续静音最多保留 10 个下标，超出部分丢弃。遍历范围恒为 `range(len_)`（`len_` 已截到首个 stop）。
- `isfix` 是 **batch 级**：任一行触发收缩，整个 batch 以 `stop_mel_token` 作 padding 值按最长行重拼（`rebuilt=True`）；未触发则 `out` 为各行的浅拷贝（`rebuilt=False`）。
- `max_len = max(code_lens)`；`T = len(codes[0])`。无论 `isfix` 取值如何，只要 `max_len < T`，输出都按 `[:max_len]` **水平切齐**（官方在 `isfix=False` 时也做这一步）。

### 3. mel_length_ledger：生成后长度账

输入 `codes, stop_mel_token, max_mel_tokens=1500, ratio=1.72`。返回 `{"code_lens", "max_code_len", "truncated_codes", "has_warned", "target_lengths", "total_frames", "ratio", "max_mel_tokens"}`。

```
has_warned = any(len(c) > 0 and c[-1] != stop_mel_token for c in codes)
code_lens = [c.index(stop) if stop in c else len(c) for c in codes]
max_code_len = max(code_lens) if code_lens else 0
truncated_codes = [c[:max_code_len] for c in codes]
target_lengths = [int(cl * ratio) for cl in code_lens]
total_frames = sum(target_lengths)
```

`has_warned` 对应官方「exceeding max_mel_tokens」告警的布尔判定（本题只判布尔，不真发 warn）；`target_lengths` 用 `int` **截断**而非 `round`（`1 * 1.72 -> 1`，不是 `2`）；`ratio` 与 `max_mel_tokens` 原样回显。

## 输入与返回示例

```
solve(mode="split_segments", tokenized=["a", "b", "c", "."],
      max_text_tokens_per_segment=120, split_tokens=["."],
      quick_streaming_tokens=0)
# {"segments": [["a", "b", "c", "."]],
#  "pre_merge": [["a", "b", "c", "."]], "total_token": 4,
#  "n_segments": 1, "merge_flags": ["first"]}

solve(mode="mel_length_ledger", codes=[[3, 4, 999, 8], [5, 6, 7]],
      stop_mel_token=999, ratio=1.72, max_mel_tokens=1500)
# {"code_lens": [2, 3], "max_code_len": 3,
#  "truncated_codes": [[3, 4, 999], [5, 6, 7]], "has_warned": True,
#  "target_lengths": [3, 5], "total_frames": 8,
#  "ratio": 1.72, "max_mel_tokens": 1500}
```

## 不进判题域

真实 BPE / tokenizer、`s2mel` 的 length regulator 帧扩展（= FastSpeech2 length-reg）、GPT 采样与 emovec、BigVGAN / mel / 波形 IO、训练式时长预测器与 latent 插值、`warnings.warn` 的真实发出、`torch.tensor` 数值前向。判题只考上述三段确定性账。
