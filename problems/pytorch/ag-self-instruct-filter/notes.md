# notes（ag-self-instruct-filter）

## 回源

- 主源：`yizhongw/self-instruct` 的 `self_instruct/bootstrap_instructions.py`（main 分支，经 GitHub contents API 取原文核对）
  - `post_process_gpt3_response`(41-70)：`response is None or finish_reason == "length"` 直接返回 `[]`；
    44 行 `re.split(r"\n\d+\s?\. ", ...)`；47-48 行 `\s+` 压缩 + `strip()` + `strip().capitalize()`；
    52 行 `len(inst.split()) <= 3 or > 150`；55 行 12 个关键词 `\b(w)\b` IGNORECASE；
    61 行 `startswith("Write a program")`；64 行 `inst[0] in string.punctuation`；67 行 `not inst[0].isascii()`。
  - 195-213 行：`max(rouge_scores) > 0.7` 丢弃（严格大于）、`all_instructions = seed + machine` 之后
    `np.argsort(rouge_scores)[-10:][::-1]` 取 most_similar、`avg = np.mean(rouge_scores)`、接受后
    `machine_instructions.append(inst)` 使池子增长。
  - `encode_prompt`(19-29)：头部两句 + `f"{idx+1}. {instruction}\n"` + 末尾 `f"{len+1}."`，
    每条先 `\s+` 压缩、`strip()`、`rstrip(":")`。
- 官方 `sample_machine_instructions`(32-34) 是 `random.sample`，随机性不入题；`make_gpt3_requests` 全略。

## 本题的规范化与划界

- **相似度作显式入参**：官方在循环里现算 ROUGE-L。本题不重算（`eval-rouge-score` 已覆盖 ROUGE-L 计分），
  而是要求调用方给出「第 i 条候选对当前池子」的向量，题目只考阈值判定、池子增长、top-k 与均值。
  这样做的副产品是一条硬约束：**被拒的候选不进池子**，所以后续向量的长度不增长——这是很容易写错的一处。
- **排序必须自己钉**：官方 `np.argsort` 默认 quicksort **不稳定**，同分时谁进 top-10 由实现决定。
  本题归一化为「按 `(分数, 下标)` 升序（等价稳定升序）取末 `top_k` 再反转」，
  于是同分时**下标大的排在字典前面**，可被用例钉住。
- **空池子**：官方 `max([])` 会直接抛错、`np.mean([])` 给 `nan`。本题归一化为
  「接受 + `max_similarity=None` + `avg_similarity=0.0`」，并把这条写进题面。
- `capitalize()` 的副作用是本题的核心陷阱：它先把整串小写，再让首字母大写，所以
  ① `WRITE A PYTHON SCRIPT TO COUNT PRIMES` 的产物是 `Write a python script to count primes`；
  ② `Write A Program To Print Hello` 会被 `startswith("Write a program")` 拦下（换成 `WRITE A PROGRAM ...`
  也一样）——这条只有把 capitalize 放在 startswith 之前才成立。
- `\b` 与 CJK 的交互实测：`file系统` 不命中 `\bfile\b`（两侧都要词边界），`profile` 不命中 `\bfile\b`，
  但 `Go  To the market` 会因为 `\s+` 压缩先变成 `Go To the market` 而命中 `\bgo to\b`（大小写不敏感）。
- 切分正则的 `\s?` 同样只认**一个**空白：`\n9  . x` 不切、`\n9\n. x` 切（换行也算一个空白）、
  `\n9.x` 不切（点后必须有一个空格）。这三条是 `m-split-star` / `m-split-no-space` 的抓点。

## 变异

28 条全部 CAUGHT（`.muts-si.json`，跑完即删）：

- 切分层：`\s?`→`\s*`、去掉点后空格；
- 归一化层：不压缩空白、不 capitalize；
- 过滤层：`<=3`→`<3`、`>150`→`>151`、删掉关键词 `go to`、`\b`→无边界、
  标点门失效、ASCII 门失效、`startswith` 改小写（capitalize 后永不命中）；
- 截断层：忽略 `finish_reason`、忽略 `response is None`；
- 去重层：`>`→`>=`、彻底不判阈值、接受后不入池、`sims` 条数校验失效；
- top-k 层：不反转（升序）、不切 `top_k`；
- 统计层：`avg` 用 `max` 顶替、空池给 `nan`；
- 提示词层：不剥尾部冒号、不压缩空白、头部恒用分类版、编号从 0 起、末行少 1。

**已知等价（未入清单）**：把 top-k 的排序键从 `(vec[i], i)` 改成 `vec[i]`——Python 的 `sorted` 本身稳定，
同分仍按下标升序，语义完全一致，故不值得作为变异。

首轮唯一 SURVIVED 是 `m-split-star`：原用例只有「一个空格 / 一个制表符」的切分样例，
`\s?`→`\s*` 无从暴露；补上「数字与点之间两个空白不切」与「换行充当那个空白仍切」两条后抓住。

## 判题注意

- 容差：`avg_similarity` 用 1e-12 比较，分数本身保持精确相等（都是入参直接搬运）。
- `most_similar` 是**有序字典**，断言时要比 `list(dict)` 的顺序，不能只比键集合；
  这也是唯一能区分「升序 / 降序 / 同分方向」的观测面。
- `reason` 的字符串是判据（`empty` / `too_short` / `too_long` / `keyword` / `write_a_program` /
  `punct` / `non_ascii`），只断言 `kept` 会丢掉「被哪道门拦下」的全部信息。
