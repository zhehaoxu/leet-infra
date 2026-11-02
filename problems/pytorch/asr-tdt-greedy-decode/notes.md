# notes — asr-tdt-greedy-decode（order 717）

## 回源

- 权威源：NVIDIA/NeMo `nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py`（GreedyTDTInfer.torch_impl）；论文 arXiv 2304.06795（TDT: Token-and-Duration Transducer）。
- **blank 强制 duration≥1**：`durations = durations.masked_fill_((durations == 0) & blank_mask, 1)`——只对 blank 施加；非 blank 的 0 duration 原样保留（进 max_symbols 分支）。
- **跳帧语义**：非 blank 发射后 `time_indices += duration`，无 RNN-T 的 blank 收尾；官方 torch_impl 有注释 *NB: difference between RNN-T and TDT*（RNN-T 发射后原地等 blank）。duration 可把时间推过句末——发射账在跳帧前快照（time_indices_current_labels），所以"跳出界"的那次发射仍计入。
- **max_symbols 强制 blank 的进帧量**：`time_indices += 1`——只进 1 帧、忽略预测 duration。不是"duration=1"的语义（duration=1 与强制 blank 的区别在 forced_blanks 账与后续帧的 logits 消费）。
- **官方无死循环防护**：非 blank duration=0 且 max_symbols=None 时 torch_impl 原地无限循环。本题加域约束 ValueError；solution 另加防御步数上限 `max_steps = 100*(T+1)`（官方无此防护，仅病理输入触发，防测试 harness 挂死）。
- 无状态化：predictor（StatelessTransducerPredictor）不建模——torch_impl 的 label loop 每步都重算 joint，predictor 状态在无状态配置（conformer_tdt_bpe_stateless）下不依赖历史，故按帧显式入参等价。

## 测试设计

- oracle = torch_impl 标量转录，五键全整数精确比对。
- argmax 恰定构造：`_rows(spec, T, width)` 每帧 winner 恰为 spec 指定下标、其余 -9——**必须显式逐帧行，用 `spec[t % len(spec)]` 循环复用与手算预期不符（首版踩过）**。
- max_symbols=1 的语义锚点：每个非 blank 后立即强制 blank → forced_blanks == 非 blank 数；与 max_symbols=2（允许连续 2 发射后强制）区分。
- 病理输入（非 blank d0 + max_symbols=None）断言 ValueError——domain-no-guard 变异的抓手。

## 变异（10 全抓）

- blank-force-off（去掉 masked_fill）：blank d0 原地停留 → 步数上限外仍可辨（blank 无发射，final_t/forced_blanks 不变但循环卡死被上限暴露）。**注意：标量 blank d0 设计下该变异必死循环，靠 max_steps 兜底而非断言差异**。
- nonblank-no-advance / force-adds-duration（强制 blank 改 `t += dur`）/ lasts-no-reset（换时间戳不清 lasts）/ emit-after-advance（时间戳改跳帧后）/ token-argmax-last（并列取末）/ emit-blanks（发射 blank）/ domain-no-guard / final-clamped（min(T,·)）/ terminal-lt-le（`<` vs `<=` 出界终止）。
- **死循环变异教训**：无超时机制的变异 harness 下，`while True:` 类变异会 SIGKILL 整个进程（exit 137）。两类死循环源：① 变异把 `t += dur` 改成 `t += 1` 之外的停留写法；② blank d0 本身无防护。solution 的 max_steps 兜底使这些变异可测（超限 RuntimeError 与 oracle 行为可判）而非挂死。

## 坑

- `_rows` 按 `t % len(spec)` 循环构造与手算预期不符 → 改显式逐帧行。
- max_symbols=1 语义：不是"最多 1 个非 blank 后结束"，是"每个非 blank 后都强制 blank"。
- 与 700（rnn-t 贪心）/699（transducer loss）划界：本题管 TDT 双头跳帧账；asr 合集 9 题至此覆盖 Whisper（DTW 对齐/语言 ID）+ FunASR（VAD）+ NeMo（TDT）三线。
