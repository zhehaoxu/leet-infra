# zip2zip LZW 动态码本编码

## 题目描述

zip2zip（2610.13605）的核心观察：LLM 的输入 id 序列里有大量重复短语，但词表在训练后就冻结了，没法把新出现的短语收进词表。zip2zip 的办法是给每条输入**现挂一本 LZW 码本**：遇到「当前 buffer 拼上一个 token」构成的短语不在码本里时，把它作为超词（hyper id）插进一本**动态码本**，发射 buffer 里已知的部分；码本不落盘、跟着这条输入走（推理时由 Rust codec 在 CPU 上现编码现解码）。这样重复模式第二次出现时就能用一个 id 发射，序列显著变短。

真正写 codec 时容易栽在三处细节上：一是 **buffer 何时发射、何时继续拼**——未知短语入本后要「弹出发射、再用这个 token 重开 buffer」；二是 **上限的三个独立开关**——码本满（`next_id` 撞上 `initial_vocab_size + max_codebook_size`）后不再生长、逐 token 发射，短语拼到 `max_subtokens` 强制发射，`truncation` 到 `max_length` 连输入都不再消费（码本只按已消费输入生长）；三是 **disabled id 的打断语义**——被禁用的 token 不参与任何合并，像分隔符一样把左右的短语切开。本题给定一条 id 序列与全部上限参数，复现这份编码，交出压缩序列、attention mask 和码本。

## 任务

实现 `solve(ids, initial_vocab_size, max_codebook_size, max_subtokens, pad_token_id, disabled_ids=None, truncation=False, max_length=None, padding="do_not_pad")`：

- `ids`：非空 `list[int]`，每个元素是非负整数且 **小于 `initial_vocab_size`**（只编码基词表内的 id）；
- `initial_vocab_size`：$\ge 1$ 的整数，基词表大小；超词 id 从它起编号；
- `max_codebook_size`：$\ge 1$ 的整数，码本最多容纳的短语数；`next_id` 的上限是 `initial_vocab_size + max_codebook_size`；
- `max_subtokens`：$\ge 2$ 的整数，单个短语最多含的基词 token 数；
- `pad_token_id`：整数，**必须 $\ge$ `initial_vocab_size + max_codebook_size`**（不得与任何可发射 id 重叠）；
- `disabled_ids`：`list` 或 `None`，落在基词表内的不重复 id，禁用 token 不参与合并；
- `truncation`：bool；为 `True` 时必须给 `max_length`；
- `max_length`：$\ge 1$ 的整数或 `None`；
- `padding`：只能是 `"do_not_pad"` 或 `"max_length"`；后者必须给 `max_length`。`"longest"` 属批量语义，不在本题。

返回 `dict`，键为 `compressed_ids`、`attention_mask`、`codebook`（超词 id → 短语，按超词 id 升序的 dict）。

## 精确规则

维护 `buffer`（当前正在拼的短语）、`codebook`（短语 tuple → 超词 id）、`next_id`（初值 `initial_vocab_size`）、`limit = initial_vocab_size + max_codebook_size`、输出 `compressed`。逐 token 消费 `ids`：

1. **disabled token**：先发射并清空手上的 buffer（若有），然后把这个 token **单独发射**、立即清空 buffer——它不进入任何短语。
2. **普通 token**：追加进 buffer。判 `known = len(buffer) == 1 or tuple(buffer) in codebook`（单 token 天然已知）。
3. **未知短语**：若 `next_id < limit`，把 `tuple(buffer)` 以 `next_id` 入本、`next_id += 1`；然后无论是否入本，都**弹出 buffer 末尾这个 token、发射 buffer 剩余部分**、清空 buffer、再以该 token 重开 buffer。
4. **上限发射**：若 buffer 长度达到 `max_subtokens`，发射整个 buffer 并清空（此时 buffer 一定是已知短语——每次入本后 buffer 都被重置为单 token）。
5. **收尾**：输入消费完后若 buffer 非空，发射它。
6. **truncation 早退**：`emit` 前检查 `truncation and len(compressed) >= max_length`，为真则不再发射；while 顶部同样检查，为真则**直接终止循环、不再消费输入**——码本因此只按已消费的输入生长。
7. **发射值**：`len(buffer) == 1` 发 `buffer[0]`，否则发 `codebook[tuple(buffer)]`。
8. **padding**：`"max_length"` 时结果短于 `max_length` 则**左侧**补 `pad_token_id` 到目标长度；长于（未开 truncation 又超长）抛 `ValueError`；**同一个 `max_length` 既是截断上限又是填充目标**。`attention_mask` 按 `id == pad_token_id` 置 0，其余置 1。

## 示例

```text
ids = [0,1,0,1,0,1,2]，initial_vocab_size=4，max_codebook_size=6，max_subtokens=3

i=0: buffer=[0]，单 token 已知
i=1: buffer=[0,1] 未知 → 入本 4:[0,1]，发 [0]，buffer=[1]
i=2: buffer=[1,0] 未知 → 入本 5:[1,0]，发 [1]，buffer=[0]
i=3: buffer=[0,1] 已知
i=4: buffer=[0,1,0] 未知 → 入本 6:[0,1,0]，发 [0,1]（4），buffer=[0]
i=5: buffer=[0,1] 已知
i=6: buffer=[0,1,2] 未知 → 入本 7:[0,1,2]，发 [0,1]（4），buffer=[2]
收尾：发 [2]

compressed_ids = [0,1,4,4,2]，attention_mask 全 1
codebook = {4:[0,1], 5:[1,0], 6:[0,1,0], 7:[0,1,2]}
```

变体（同输入）：

- `ids=[0,1,0,1,2,2,3,3]`、`max_codebook_size=2`：码本满在 `{4:[0,1], 5:[1,0]}`，后续短语不再入本、逐 token 发射 → `[0,1,4,2,2,3,3]`。**满码本的判据是 `next_id < limit` 失败，不是「短语不在码本里」**——`[0,1]` 已知照发超词 id，只是新短语（如 `[2,2]`）不再收编。
- `truncation=True, max_length=3`：发射到 `[0,1,4]` 封顶，while 顶部早退，码本只长到 `6:[0,1,0]`（`7:[0,1,2]` 不会入本——输入根本没被消费到）。
- `padding="max_length", max_length=8`：`[100,100,100,0,1,4,4,2]`，mask `[0,0,0,1,1,1,1,1]`。
- `disabled_ids=[1]`：1 处处单发、切断合并，`[0,1,0,1]` 不会产出短语 `[0,1]`。

## 约束

- 不修改输入；所有非法输入抛 `ValueError`（类型、范围、`pad_token_id` 重叠、`disabled_ids` 越界/重复、truncation/padding 与 `max_length` 的组合约束、未截断超长）。

## 与相邻题目的边界

- `bpe-training` / `bpe-encoding` 学的是**训练期冻结词表**的合并表；zip2zip 是**推理期现挂**的输入级码本，码本不进模型、随序列生灭。
- `data-collator` / `sequence-pack-pad` 的 padding 是批量整理；本题的 padding 是**单条编码的尾部产物**，且 `max_length` 同时管截断。
- `dt-minhash-lsh-dedup` 压的是**语料条数**（去重）；本题压的是**单条序列长度**（token 数）。
