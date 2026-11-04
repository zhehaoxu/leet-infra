# 解题笔记

## 思路

官方 ezelikman/STaR 的确定性核心（`device_inference.py`）：

1. **eval_output**（L27-91）：逐样本抽取链（cqa 首行 `[-3]` 字符 / gsm 首个 `####` 行截断 / arithmetic 行列表上下标链）+ `lower()` 判对 + 训练样本组装（base + 抽取中间串 + endoftext）+ 按 class 分桶 `{accurate, total}`；
2. **get_score**（L94-97）：`total==0 -> 0`（int）；
3. **question_to_context**（L99-124）：三数据集上下文组装（cqa 选项列表 + (CORRECT) 标注 / gsm 答案 hint / arithmetic `Target:` 相邻切片对）；
4. **eval_examples**（L207-238）+ **examples_to_batch**（L127-186，去分词）：分批评测、miss 以 `p_rationalization` 准入 hint 池、池满一批以 `key+'_r'` 重命名重评、`p_show_hint_save` 以概率把保存上下文换成 hint 版、`few_shot_train` 加 few-shot 保存前缀（arithmetic raise）。

上游不入判题域：mesh_transformer/jax 的 `network.generate` 与 numpy 批填充（输出以已解码字符串显式入参）、GPT2 tokenizer 编解码（输入上下文经 `gen_fn` 替身消费）、wandb/tqdm/checkpoint/smart_open、`iteration_train.py` 的 os.system 编排（gen_train/gen_records/train_model/eval_model）与 `get_n_steps` 增长账（steady/exponential/数数据点——纯 shell 编排无分支语义增量，不立项）。

与现有题划界：
- **775/776/777/778/779**：demo 拼装、候选评分、RoT 序列编译、CR 状态图；本题是**评测-写回-理性化**闭环——账本核心是「判对样本组装回训练集」与「miss 样本换 hint 上下文重评」两条流；
- **779 的 verifier 三态**：CR 的 verdict 驱动图生长；本题 hint 池是无状态 FIFO（`_r` 重命名 + carry-over），判对样本直接落盘不进图。

## 关键实现点（官方口径逐条钉死）

- **total 在 try 之前自增**：`accuracy[example_class]['total'] += 1` 在 `try:` 前——IndexError 跳过的短输出也计数。total==0 时 `get_score` 返回 **int 0**（非 float）。
- **cqa 抽取**：`output_numbers[0]` 只取首行 -> 剪 endoftext 子串 -> `[-3]` **字符**索引；len<3 抛 IndexError -> `except IndexError: pass` 静默跳过。多行输出第二行永远不参与。
- **gsm 抽取**：扫**首条**含 `'####'` 的行，`output_numbers = '\n'.join(lines[:line_idx+1])` 截断（保存用的就是截断串，**未 strip**），剪 endoftext，预测 `split('####')[-1].strip()`；无 `####` 时 `output_prediction=""` 判错不保存（output_numbers 保持 list，若 target 空串判对会在拼 new_example 时 TypeError——官方只捕 IndexError，本题同语义）。
- **arithmetic 抽取**：`output_numbers` 还是**行列表**——`'<|endoftext|>' in output_numbers` 与 `'</scratch>' in output_numbers` 是**整行相等**的成员测试（非子串）；预测行下标链：endoftext 行 `-1`（第 0 行时得 `-1`，Python 负索引取**最后一行**）、`</scratch>` 行 `+1`（==len 则 continue 跳过）、否则 `direct and len>1 -> 1 else 0`；保存 `'\n'.join(lines[:pi+1])` 后**再剪一次** endoftext 子串（跨行子串在这里才被剪掉）。
- **判对**：`output_prediction.lower() == target.lower()`，两侧大小写不敏感；写回 = `base_context + 抽取中间串 + endoftext`。
- **eval 批循环**：`len(main_examples) == eval_batch_size` 才评（**最后不足批永不评测**）；批评完逐 miss `rng.random() < p_rationalization`（**and 短路，只对 miss 翻币**）入池；`rationalize and len(hint_pool) >= eval_batch_size` 时立即取**前一批**（`key+'_r'` 重命名 -> accuracy 独立桶）以 `hint_prompts, hint=True` 重评，`hint_examples = hint_examples[batch:]` 剩余留池（carry-over：池 4 批 3 -> flush 3 留 1）。
- **批内上下文**：hintless 与 base（hint 版）各算一份；arithmetic 时 `cur_few_shot = base.split('\n\n')[:-1]` **覆盖**传入 few-shot（本批后续样本也用它——每样本重算）、两份取 `split('\n\n')[-1]`；`rng.random() < p_show_hint_save` 把保存用 hintless 换成 base（该怪癖只在 hint 批可观测——hint=False 时两者相同）；`few_shot_train` 保存加 `'\n\n'.join(train_prompts) + '\n\n'` 前缀（arithmetic 直接 NotImplementedError——在批准备期就 raise，不等判对）。
- **target 抽取**（examples_to_batch）：arithmetic 取 `split('\n')[-3]`（行数<3 → 'invalid'）、cqa 取 `answerKey`、gsm 取 `answer.split('#### ')[-1]`（带空格，与 eval_output 的 `'####'` 不同）。
- **翻币顺序**：批内逐样本 `p_show_hint_save` 先于批后逐 miss 的 `p_rationalization`；hint 批批内翻币同序。rng 为显式入参的 `random.Random`。

## 测试设计（21 例）

extract 6（cqa 三态含多行命中/endoftext 剪/IndexError、gsm 无 `####`/双 `####`/strip/大小写、arithmetic 六路含 `-1` 负索引与整行成员测试、direct 单行/多行、few_shot_train raise、长度不齐）、score 1（int 0 与 0.375）、context 3（cqa 含大写 answerKey 匹配、gsm hint、arithmetic 四变体含 direct+hint 与无 Target）、eval 10（leftover 永不评测、池不足不 flush、`p_show_hint_save=1.0` 的 (CORRECT) 落盘、p_rat=0、few_shot_train 前缀 + arithmetic raise、arithmetic few-shot 覆盖、`_r` 分桶、双 flush、p_rat=0.5 撒币序列（Random(7) 官方沙箱实测）、carry-over 池 4 flush 3 留 1）、args_validation 1（11 路非法入参）。全部期望值由官方源码 exec 沙箱（stub args/basic_open/tqdm/transformers/随机数、eval_batch 以 shim 镜像 examples_to_batch 非分词部分）实测钉死。

## 回源偏差记录

- 官方 `eval_output` 把训练样本 `print(..., file=new_train_f)` 写文件；本题改为返回 `train_examples` 列表（端口等价：文件内容 = 列表逐项拼接）。
- 官方 `eval_examples` 内部 `transformers.GPT2TokenizerFast` + `network.generate` + `tqdm`；本题以 `gen_fn` 替身替代（输出已解码），`pbar.set_description` 的分数展示不入判题域。
- 官方 `args.dataset_mode`/`args.few_shot_train`/`args.no_prompt` 是模块级全局；本题改为显式入参（`prompts=None` ⇔ no_prompt）。
- 官方 gsm 无 `####` 且 target 空串判对时拼 list + str 会 TypeError（官方未捕）；本题保持同语义不特判。
