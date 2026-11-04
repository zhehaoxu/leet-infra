# 踩坑记录（order 829）

## 回源口径

- 仓 `FunAudioLLM/SenseVoice`，`model.py`（SenseVoiceSmall 类）。官方仓 python 侧**没有** rich_transcription_postprocess 源码（runtime 目录只有 llama.cpp 导出与 detok）——该后处理散落在 funasr 包/社区实现里，判不进精确口径，不入题（与 QUITO 同判例：官方源码不自洽的部分不做）。
- 前缀顺序易读错：官方 inference 先 `cat(textnorm_query, speech)` 再 `cat(input_query, speech)`，嵌套后前缀是 **[lid, event, emo, textnorm]**，不是 [textnorm, lid, ...]。逐行核对两次 cat 的张量顺序才钉死。
- 训练期 `torch.rand(1) > 0.2` 的 lid 采样（20% 回落 auto）是 RNG——判题域只收 inference 侧的确定性映射。
- embed 表是 16 行小表（行号 0..15），与 CTC 词表的 24884/25016 等大 id 分离，中间隔一层 dict 映射（lid_int_dict）。两套 id 千万别混。

## 测试设计踩坑

1. **LabelSmoothingLoss 的 .float()**：官方实现内部把 logits 转 float（fp32），fp64 判题下逐位置对拍直接 dtype 冲突（"Float did not match Double"）。重建版去掉强转，保持输入 dtype。
2. **ctc_shift 偏移方向**：首版把内容区间写成 [4, L-4)（把 `lens - 4` 误解成 end 的绝对坐标）——实际 `encoder_out[:, 4:]` 配 `lens - 4` 指内容帧数，区间是 [4, L)。用 L=12 断言 8 内容帧钉死。
3. **ban_emo changed 的真等价**：`(before != after)` 与 `(before == emo_unk)` 数值上同值（屏蔽不动其他列）——变异 `changed = (before != after)` 幸存属真等价，剔除换成"屏蔽错列"变异。
4. **_th_accuracy 与 _label_smoothing_loss 的 keep 行同文**——mut 锚点必须带后继行区分（首版 count==2 被 assert 拦下）。
5. unique_consecutive 的 CTC 语义：只折叠**相邻**同 id，非相邻同 id 保留（test 用 [3,4,3] 钉死）；folded 序列里 blank 也参与折叠，再整体过滤。

## mut 设计（14 变异全 CAUGHT）

- 前缀族：顺序错 / event-emo 行号互换 / n_added 错报 —— 查表行号逐帧断言兜住；
- rich CE 族：ignore 丢失（分母污染）/ 均匀项错值（gather 成 nll）/ ε 权重互换 / acc 分母含 ignore —— ε=0、ε=1、ε 混合闭式三方对拍 + 全 ignore 边界；
- greedy 族：丢 unique_consecutive / 滤 blank 失效 —— [3,3,0,0,3,5,5,0,7] 序列钉死折叠+过滤两步；
- ban_emo 族：-inf 改 -1 / 屏蔽错列 —— -inf 断言 + argmax 永不落 unk + changed 掩码三重；
- ctc_shift 族：起点 0 / end 少 4 / max 兜底丢失 —— [12,4,5,3] 四边界用例。
