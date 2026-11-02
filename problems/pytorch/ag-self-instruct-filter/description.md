# self-instruct：合成指令过滤与相似度去重账

## 题目描述

self-instruct 用「种子指令 + 模型续写」自举出大规模指令数据，但模型吐出来的东西必须过两道闸：**规则过滤**（太短太长、涉及图像/文件、以标点开头、非英文开头……）与**相似度去重**（与已有指令的 ROUGE-L 相似度超过阈值就丢）。通过的指令进入池子，下一轮再拿池子里的样例去拼提示词。

本题把这条流水线做成纯函数：给一段模型原始输出、种子指令、以及**逐条候选对池子预计算好的相似度向量**，输出每段的过滤账、去重账、最终指令表与下一个提示词。相似度本身不在本题内（`eval-rouge-score` 已覆盖 ROUGE-L 计分），这里只按显式入参做阈值判定。

## 规则

签名 `solve(response, seed_instructions, sims, finish_reason="stop", threshold=0.7, top_k=10, prompt_instructions=None, classification=False)`。

**1. 截断与切分**。`response is None` 或 `finish_reason == "length"` 时直接得到空结果（模型被截断，官方整批丢弃）。否则按 `\n\d+\s?\. ` 切段：换行 + 数字 + **至多一个**空白 + 点 + **一个空格**。所以 `\n9. x`、`\n9.  x`（点后两空格）、`\n9 . x`、`\n9\t. x`、`\n9\n. x` 都切；`\n9.x`（点后无空格）与 `\n9  . x`（数字与点之间两个空白）不切。文首没有该前导的那一段**照样保留**并参与过滤。

**2. 归一化**。`\s+` 压成单空格 → `strip()` → `capitalize()`。`capitalize()` 会把除首字母外的**全部字符转小写**，因此 `WRITE A PYTHON SCRIPT` 变成 `Write a python script`，而 `Write A Program To ...` 变成 `Write a program to ...`。

**3. 七道过滤门**（按官方顺序，给出第一条命中的理由）：

- `empty`：归一化后为空串；
- `too_short`：`len(inst.split()) <= 3`；
- `too_long`：`len(inst.split()) > 150`；
- `keyword`：12 个词 `image/images/graph/graphs/picture/pictures/file/files/map/maps/draw/plot/go to` 任一个按 `\b(word)\b` 大小写不敏感命中；
- `write_a_program`：以 `"Write a program"` 开头（**在 capitalize 之后**判，所以 `Write A Program` 也会被拦）；
- `punct`：首字符属于 `string.punctuation`；
- `non_ascii`：首字符不是 ASCII。

注意 `\b` 是正则词边界：`profile` 不命中 `file`；`file系统` 里 `file` 后紧跟 CJK 字符、**没有**词边界，因此也不命中。`go  to`（两个空格）会先被归一化压成 `go to` 再命中。

**4. 阈值去重**。`sims[i]` 是第 `i` 条通过第一层的候选对当前池子的相似度向量，池子 = 种子指令 + **已被接受**的机器指令（被拒的不入池）。因此第 `i` 条的向量长度必须等于当时的池大小，否则抛 `ValueError`。

- 向量非空且 `max(vec) > threshold` → 丢弃（**严格大于**，等于阈值算通过）；
- 记为接受时把该指令追加进池子；
- `avg_similarity` 是向量均值；池子为空时 `max_similarity` 记 `None`、`avg_similarity` 记 `0.0`（不做除法）。

`sims=None` 表示不提供相似度：跳过去重层，第一层通过的候选全部按顺序接受，`max_similarity=None`、`avg_similarity=0.0`、`most_similar={}`。

**5. 最相似项**。仅对**被接受**的候选给出 `most_similar`：把向量按 `(分数, 下标)` 升序排序（等价于稳定升序），取末尾 `top_k` 个再反转，得到「分数降序、同分下标大者在前」的字典；池子不足 `top_k` 时全给。

**6. 提示词拼装**（`prompt_instructions` 非 `None` 时）。头部两选一：`classification=True` 用 `Come up with a series of classification tasks. Try to specify the possible output labels when possible.`，否则 `Come up with a series of tasks:`；随后每条一行 `{i+1}. {text}`，`text` 做 `\s+` 压缩、`strip()`、`rstrip(":")`（只剥**尾部**冒号，且可能留下冒号后的空格）；最后一行是 `{n+1}.` 且**不带换行**。

## 示例

```text
输入：response = "Here are some tasks:\n1. Write a short poem about the sea\n2. be nice\n"
      seed_instructions = ["seed one"], sims = None
输出：segments = [{"raw": "Here are some tasks:", "instruction": "Here are some tasks:",
                  "kept": true, "reason": null},
                 {"raw": " Write a short poem about the sea", ...},
                 {"raw": "be nice\n", "instruction": "Be nice", "kept": false,
                  "reason": "too_short"}]
      candidates  = ["Here are some tasks:", "Write a short poem about the sea"]
      instructions = 同上（sims=None 全部接受）
      prompt = null
```

```text
输入：prompt_instructions = ["Summarize   this   text", "Rate it:", "Weird: :"]
输出：prompt = "Come up with a series of tasks:\n1. Summarize this text\n2. Rate it\n3. Weird: \n4."
```

## 约束

- `response` 是字符串或 `None`；`finish_reason` 是字符串；`seed_instructions` 是字符串列表（可为空）；`sims` 是「实数列表」的列表或 `None`，实数不得含 `nan` / `inf`。
- `threshold` 必须是 `[0, 1]` 内有限实数且拒收 `bool`；`top_k` 必须是 `>=1` 的 `int`（拒收 `bool`）；`classification` 必须是 `bool`。
- `sims` 条数必须等于第一层通过的候选数；每条长度必须等于当时的池大小。
- 返回固定字段 `segments` / `candidates` / `dedup` / `instructions` / `prompt`；`segments` 每条含 `raw` / `instruction` / `kept` / `reason`，`dedup` 每条含 `instruction` / `accepted` / `max_similarity` / `avg_similarity` / `most_similar`。
- 入参不得被就地改写（种子池按值复制再增长）。
