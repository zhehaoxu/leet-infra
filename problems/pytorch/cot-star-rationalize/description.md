# STaR：自举理性化账（三数据集抽取 + 答对写回 + hint 池理性化）

## 题意

STaR（arXiv:2203.14465, 官方仓 `ezelikman/STaR`）的自举循环里，`device_inference.py` 把「生成 -> 抽取答案 -> 判对错 -> 答对样本写回训练集」组织成两个确定性账本：`eval_output`（L27-91）按数据集抽答案、判对错、组装新训练样本并按 class 分桶计数；`eval_examples`（L207-238）把数据集按 `eval_batch_size` 分批，答错样本以概率进 hint 池、池满一批就以 `key+'_r'` 重命名用带提示上下文重评（理性化）。本题把 LLM 生成全部显式入参（`outputs` 已解码字符串 / `gen_fn` 可调用替身），考纯账本：

```text
mode="extract": outputs + answers + base_contexts + classes + dataset_mode
                -> {"successful", "train_examples", "accuracy"}
mode="score":   counts {"accurate","total"} -> total==0 时 int 0，否则 accurate/total
mode="context": example + dataset_mode (+hint/direct) -> 上下文字符串
mode="eval":    dataset + prompts/hint_prompts + gen_fn + rng + eval_batch_size
                (+rationalize/p_rationalization/p_show_hint_save/few_shot_train)
                -> {"accuracy", "train_examples"}
```

## 账本规则（官方逐条钉死）

`mode="extract"`（官方 `eval_output`；**total 在抽取 try 之前自增**，抽取抛 IndexError 被 `except IndexError: pass` 静默跳过——样本计数但不写回）：

```text
output_numbers = 输出.split('\n')；accuracy[class] 缺则初始化 {'accurate':0,'total':0}，total += 1
cqa:         取首行 -> 剪 '<|endoftext|>' 子串 -> 预测 = 首行[-3] 字符（len<3 -> IndexError -> 跳过）
gsm:         扫到首条含 '####' 的行 -> output_numbers 截断到该行（含）-> 剪 endoftext
             -> 预测 = split('####')[-1].strip()；扫不到 '####' 预测空串（判错不保存）
arithmetic:  output_numbers 仍是行列表 —— '<|endoftext|>' 与 '</scratch>' 都是
             **整行相等**的成员测试：
               endoftext 行下标 -1 为预测行（下标 0 时得 -1 -> Python 负索引取最后一行）
               '</scratch>' 行下标 +1；若 == len(output_numbers) -> 跳过该样本
               否则 direct 且行数>1 取下标 1、否则 0
判对:        预测剪 endoftext 子串后与 target 两侧 .lower() 比较
写回:        cqa/gsm: new_example = base_context + output_numbers + endoftext
             arithmetic: join(行[:预测行+1]) -> 剪 endoftext -> base + join + endoftext
             few_shot_train 且 arithmetic 且判对 -> NotImplementedError（官方 raise）
```

`mode="eval"`（官方 `eval_examples` + `eval_batch`/`examples_to_batch` 去分词部分）：

```text
逐样本入 main；凑满 eval_batch_size 评一批（不足的最后一批**永不评测**）；
批评完后逐样本：不在 successful 且 rng.random() < p_rationalization -> 进 hint 池；
rationalize 且池 >= eval_batch_size -> 立即取前 eval_batch_size 个
（key 追加 '_r' -> accuracy 独立桶）以 hint_prompts、hint=True 重评，剩余留池。
批内逐样本：
  hintless（无提示）与 base（hint 版）上下文各算一份；
  arithmetic: few-shot 改取 base.split('\n\n')[:-1]，两份取 split('\n\n')[-1]
  rng.random() < p_show_hint_save -> 保存用 hintless 换成 base
  few_shot_train: 保存用 '\n\n'.join(train_prompts) + '\n\n' 前缀
                  （arithmetic 直接 NotImplementedError）
  输入上下文 = '\n\n'.join(few_shot) + '\n\n'（prompts=None 则空）+ base
  target: arithmetic 取倒数第三行（行数<3 为 'invalid'）、cqa 取 answerKey、
          gsm 取 answer.split('#### ')[-1]
```

`mode="context"`（官方 `question_to_context`）三式见 solution 文档串。`mode="score"`：`total==0` 返回 **int 0**。

## 关键怪癖（官方口径）

- cqa 的 `[-3]` 是**字符**索引——只看首行，endoftext 先剪再取；
- gsm 保存的是**未 strip 的截断串**（strip 只作用于预测）；
- arithmetic 的 endoftext/`</scratch>` 成员测试是**整行相等**（跨行子串不算），endoftext 在第 0 行时 `-1` 倒着取最后一行；
- arithmetic 保存用 join 到预测行（含），再剪一次 endoftext 子串；
- hint 批的 class 是 `key+'_r'`，与主批分开计数；保存用 hintless（`p_show_hint_save` 翻中才换 hint 版）；
- `p_rationalization` 翻硬币**只对 miss 样本调用**（and 短路），顺序在批内 `p_show_hint_save` 之后。

## 输入约定

- `outputs`：每样本已解码的模型输出原文；`answers`/`base_contexts`/`classes` 与之对齐；
- `dataset`：`(key, example)` 对列表；cqa/gsm 样本是 dict、arithmetic 是原文串；
- `gen_fn(contexts, hint) -> list[str]`：确定性生成替身；
- `rng`：`random.Random` 实例（翻硬币顺序见上）。
