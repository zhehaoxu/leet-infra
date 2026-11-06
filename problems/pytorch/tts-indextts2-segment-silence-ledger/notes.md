# 踩坑与回源记录（order 835）

## 回源口径

- 仓 `index-tts/index-tts`，默认分支 **`main`**（不是 master；833 期踩过 404）。用 GitHub API `contents?ref=main` + `base64.b64decode` 下载（`raw.githubusercontent.com` 在 833 期已证实不稳），关键源码落到 `/tmp/it_front.py`、`/tmp/it_infer_v2.py`、`/tmp/it_length_regulator.py`。

```
indextts/utils/front.py        TextTokenizer.split_segments_by_token / split_segments / punctuation_marks_tokens
indextts/infer_v2.py           remove_long_silence / generation-loop code_lens / target_lengths
indextts/s2mel/modules/length_regulator.py    (out of scope)
```

- WebSearch 里的「毫秒级时长控制」多为营销/二手描述，目标 token 预测器、latent 插值是训练式模型内部机制，确定性差，**不进题**；改取仓库里三段纯序列/整数确定性账立题。
- 回源对拍脚本 `/tmp/verify835.py` 用 `ast.get_source_segment` 抽官方 `split_segments_by_token` 原样 `exec`（stub `warnings.warn` 为记录器、`List` 从 typing 取），`remove_long_silence` 直接调官方方法（传 `.venv-eval` 里的 torch tensor），`target_lengths` 按官方 `(code_lens * 1.72).long()` 对齐：用 `.venv-eval/bin/python` 运行（`.venv-backend` 无 numpy/torch）。**split 720 + remove 30 + mel 18 = 768 组全一致**。

## 机制要点

- **`split_segments_by_token` 的四路优先级**：逗号递归 > 连字符递归 > 未超限（`cur_len <= max_seg`，其中 `token in split_tokens and cur_len > 2` 才切）> 超限硬切。前两路把 `cur` 用新 `split_tokens` 递归重跑，递归里 `,`/`-` 已在 `split_tokens` 中故不会再次触发。
- **官方 `for` 循环 `i += 1` 不生效**：切段时把引号 token 追加进 `cur` 后写 `i += 1`，但 `for i in range(...)` 下一轮仍会处理该引号，故引号 token 会双重出现（手算锚点 3 钉死）。本题实现如实复现，不「修 bug」。
- **合并状态机**：`total_token` 是跨段累计（不是当前段长度）；`max_seg / 2` 是浮点除法；`quick=0` 时 `total_token > quick` 几乎恒真，`merge_half` 分支才体现 `max_seg/2` 阈值；`merge_flags` 逐段暴露分支选择。
- **连字符递归的可观测性**：外层在第一个 `-` 被 append 后立即触发递归，故多数输入下递归切分退化成「把 `cur` 当整段」（`-` 在末尾且 `cur_len` 不超 `max_seg`）。只有当 `-` 恰使 `cur_len > max_seg` 时，递归内部走硬切、产生两段，才与「整段」可区分（手算锚点 7）。
- **`remove_long_silence` 三个坑**：`count` 是整行全局计数（含 stop 之后），多段短静音也会触发；`n` 遇非静音归零，连续静音最多留 10；`isfix` 是 batch 级，且 **`isfix=False` 时官方仍做 `codes[:, :max_len]` 水平切齐**（初读 242–291 行易漏，须读 292–297 行）。
- **`target_lengths = (code_lens * 1.72).long()`**：`int` 截断不是 `round`，`1 * 1.72 = 1` 而非 `2`；`has_warned` 只判 `c[-1] != stop` 的布尔，不真发 warn。

## 测试设计

- 3 个 `check_*`：`check_split_segments` / `check_remove_long_silence` / `check_mel_length_ledger`；judge 双模式 `run` 与 `submit` 均 `3/3 ACCEPTED`。
- 独立 oracle 用纯 Python 重写，**不 import solution**；每个 mode 做全网格对拍 + 手算锚点（手算锚点独立于 oracle 与 solution）。`split` 网格 11 个 token 序列 × 3 个 `max_seg` × 3 个 `quick` × 4 个 `split_tokens`；`remove` 6 个 codes × 4 组参数；`mel` 5 个 codes × 3 组参数。
- 中间量必须暴露：`pre_merge` / `merge_flags` / `total_token` / `n_segments`、`counts` / `kept_indices` / `isfix` / `rebuilt` / `max_len`、`code_lens` / `truncated_codes` / `target_lengths` / `total_frames` / `has_warned` 全部进返回值，否则分支顺序类与收缩类变异不可观测。
- **首轮 mutation 暴露的两个覆盖缺口及补法**：① 默认 `split_tokens` 的用例里 `"!"` 位于 `cur_len=2` 未触发切分，删掉默认常量中的 `"!"` 无法察觉 —— 把默认路径用例改成 `"!"` 在 `cur_len=4` 触发，并加手算锚点 6；② 现有含 `-` 用例里连字符递归恰好等价于整段，`sub = [cur]` 变异存活 —— 补 `["a","b","c","-"]` 且 `max_seg=3`（递归内部硬切成两段）的 grid 条目与手算锚点 7。补完 `38/38 CAUGHT, 0 SURVIVED`。
- 另补锚点 5（连续静音段被非静音打断时 `n` 归零），使 `remove-n-reset` 变异死于断言而非形状不变。

## mutation 设计（38 变异全 CAUGHT）

- split 族（18）：默认标点集缺项、空输入 total、逗号变体漏、逗号/连字符递归退化、`-` 计数阈值、`cur_len > 2` 漏、引号吞并、硬切块大小/步长、`total_token > quick` 反、`max_seg/2` 放宽、`total_token` 重置、first/new flag 互换、`pre_merge` / `n_segments` 置空；
- remove 族（12）：stop 索引、count 取错/只算 `len_` 内、触发阈值、`n < 10` 改 `< 11`、`n = 0` 重置失效、`isfix` 置反、pad 值、clip 失效、`kept_indices` 清空、`rebuilt` 置反、`max_len` 置零；
- mel 族（8）：`has_warned` 条件反、`code_lens` 不截 stop、`max_code_len` 置零、不切齐、`int` 改 `round`、`total_frames` 置零、`ratio` 写死 / 回显置零、`max_mel_tokens` 置零。
- 真等价变异先验剔除：`segments.extend(sub)` → `+=`（同列表语义）；`len_ = stop_idx[0] if ... else T` 的 else 分支由「有/无 stop」两组用例覆盖，不构造不可达分支。
- 锚点字符串必须精确匹配源码换行/缩进（834 期 `select-suffix-no-dot` 的教训）；本轮 38 个锚点 `SRC.count == 1` 全部命中。

## 查重划界

- `tts-fastspeech2-length-reg`：phoneme duration → 帧**重复扩展**；本题 `target_lengths` 是 mel code 数 × 1.72 的**整数换算**，core 是分段 + 静音收缩，不重复。
- `tts-mas-monotonic-alignment` / `tts-rvq-residual-quant` / `tts-zh-tn-num-rules`：对齐搜索 / 残差量化 / 中文 TN 数字规则，均无序列收缩或按标点递归分段。
- `data-tokenization` 侧 `sequence-pack-pad`（拼接填充）、`dt-protected-truncation`（保护式截断）、`dt-gopher-rules`（过滤规则）、`sp-unigram-segmentation`（unigram 分词概率）：**无「按标点优先级递归切分 + 合并状态机」**；`split_segments` 是 TTS 长文本按语义标点分段，机制独立。
- `asr-fsmn-vad-endpoint`（VAD 端点）、`asr-whisper-quality-gate`（质量门）、`asr-moonshine-stream-verify`（流式校验）里的「静音」是端点检测/质量词，与「TTS 生成 codes 里连续静音 token 收缩」不同。
- 红线：真音频 IO / mel / 波形 / librosa / soundfile 不出；训练式时长预测器、latent 插值、GPT 采样不出；`warnings.warn` 侧只判布尔；参数全作显式入参。
