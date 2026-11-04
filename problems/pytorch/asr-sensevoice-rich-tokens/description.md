# SenseVoice rich-token 多任务前缀账

## 背景

SenseVoiceSmall（FunAudioLLM/SenseVoice，`model.py`）用一个 CTC 头同时出四类输出：语音内容 token、语言 id、事件/情绪 token、文本规范标记。做法是**在 CTC 输入前拼 4 帧 rich-token query**，前 4 帧的预测按 rich CE 训练，第 5 帧起才是真正的语音 CTC。本题重建其中可判定的账目。

## 机制拆解

### 1. 前缀查询构造（encode_prefix）

16 行小 embed 表与官方常量：

```
embed 行：0=auto, 1=event, 2=emo, 3/4/7/11/12/13=zh/en/yue/ja/ko/nospeech, 14/15=withitn/woitn
LID_INT_DICT = {24884:3, 24885:4, 24888:7, 24892:11, 24896:12, 24992:13}
TEXTNORM_INT_DICT = {25016:14, 25017:15}
EMO_DICT = {unk:25009, happy:25001, sad:25002, angry:25003, neutral:25004}
```

inference 拼接顺序（读官方代码两次 `torch.cat` 的嵌套）：先 `speech = cat(textnorm, speech)`，再 `speech = cat((language, [event, emo]), speech)`——**最终前缀是 `[lid, event, emo, textnorm]`**，`speech_lengths += 4`。训练 encode 同序（style/textnorm 查 `text[:, 3]`，lid 查 `text[:, 0]`，训练期 lid 有 20% 概率回落 auto——该 RNG 分支不进判题域）。

### 2. rich CE（rich_ce）

`_calc_rich_ce_loss`：前 4 帧 encoder 输出经 `ctc.ctc_lo` 投影到 vocab，标签是前 4 个 rich token id，损失为 LabelSmoothingLoss（Kaldi 风格：`(1-ε)·nll + ε·(-mean logp)`，ignore_id 位置不计分子也不计分母，非 ignore 位置数归一），准确率为 `th_accuracy`（ignore 位置不计分母）。

### 3. CTC greedy（ctc_greedy）

`log_softmax → 逐帧 argmax → torch.unique_consecutive（同 id 连帧折叠）→ 过滤 blank`。frames 保留折叠前的 argmax 序列。非相邻的同 id 不折叠（是 CTC 而非全量去重）。

### 4. ban_emo_unk（ban_emo）

`ctc_logits[:, :, emo_unk] = -inf`。屏蔽后该位置永不被 argmax 选中；`changed` 掩码恰为 `before == emo_unk`（before≠emo_unk 的帧屏蔽不动其他列、argmax 不变——这一等价性使"changed=before!=after"与之同值）。

### 5. CTC 偏移窗（ctc_shift）

内容 CTC 用 `encoder_out[:, 4:]` 配 `encoder_out_lens - 4`，即原序列帧区间 `[4, lens)`（内容帧数 lens-4）；lens<=4 时为空窗。

## 判题域

- embed 权重、ctc_lo、logits 全部显式入参，fp64，无 RNG（训练期 20% lid 回落不进域）；
- 5 mode：encode_prefix、rich_ce、ctc_greedy、ban_emo、ctc_shift；
- 不出：encoder 前向、Specaug/CMVN 前端、ctc_forced_align 时间戳（依赖 funasr 工具）、rich_transcription_postprocess 文本清理（官方仓 python 侧无源码，散落在 funasr 包内）。
