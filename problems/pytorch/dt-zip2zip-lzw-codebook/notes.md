# 参考做法：LZW 生长期、三道上限与 disabled 打断

## 实现

1. 校验先行且**标量先于序列**：`initial_vocab_size/max_codebook_size/max_subtokens/pad_token_id` 逐个 `int`（拒 bool）与下界，`pad_token_id >= initial + max_codebook`；然后逐个查 `ids` 的类型与范围（**上界 `< initial_vocab_size` 容易漏**——只查非负会让 `[0, 4]` 溜进来）、`disabled_ids` 落基词表内且不重复、`truncation` 是 bool、`padding` 白名单、truncation/max_length 组合约束。最后还有一条组合约束：`padding="max_length"` 且结果超长抛 `ValueError`。
2. **主循环五态**：disabled 打断（flush + 单发 + 清空）、buffer push、known 判定（单 token 天然已知）、未知入本 + **弹尾发射重开**、`max_subtokens` 满发射清空。入本后必须 `buffer.pop(); emit(); clear(); append(token)`——直接 `emit()` 整个 buffer 会把未收编的新短语也发出去，错一格后面全错。
3. **上限三道独立**：
   - 码本满：`next_id < limit` 才入本；满了不抛错，短语**逐 token 发射**（已知短语照发超词 id）。
   - `max_subtokens`：buffer 拼到上限强制发射——保证输出短语长度 $\le$ `max_subtokens`。
   - `truncation`：`emit` 与 while 顶双检查 `len(compressed) >= max_length`；while 顶命中时**直接 break、不再消费输入**，所以码本只按已消费输入生长——这是 truncation 用例的考点。
4. **padding 与 truncation 共用一个 `max_length`**：它是截断上限，也是左填充目标；「截断到恰好 max_length 时填充为空操作」。超长未截断在 padding 阶段抛错。
5. 输出：`attention_mask` 按 pad 判 0；`codebook` 按超词 id 升序重排为 `{hyper: list(phrase)}`。

## 正确性检验

7 个 `check_*` 覆盖：题面示例（手算 LZW 全生长期）、5 组固定数据对拍（满码本、disabled 打断、`max_subtokens=2` 紧凑合并、长重复块撞上限、全 disabled）+ **往返校验**、truncation 两种（含 `max_length=1` 发射一条即停）、padding 四种形态（左填充、截断+填充共用 max_length、恰好封顶、恰好等长）、码本冻结、`max_subtokens` 上限（`[0]*6` → `[0,4,4,0]`）、18 条非法输入。

**往返校验**是本题 oracle 的核心：`test_cases.py::_oracle_decode` 独立移植上游 `decode_fn`（KwKwK 自指合并——未知超词解码时 `buffer.append(buffer[0])` 后**用带进来的 id** 入本、buffer 满时 force-merge、码本满后 `buffer.clear()` 冻结、while-merge 更新相），在无 truncation、无 padding 的用例上断言 `decode(encode(ids)) == ids`。编码侧任何一处发射/入本次序错误都会让往返炸掉，比单纯对拍压缩序列更有说服力。`_verify` 另有结构断言：mask 与 pad 一致、超词 id 从 `initial_vocab_size` 起连续、短语长度与范围合法、输入不被修改。

变异测试（8/8 全抓）：`next_id` 不自增、入本与发射次序对调、`max_subtokens` 满不发射、disabled 前 flush 丢失、左填充改右填充、truncation 不早退（码本多长一级）、超长不抛错、单 token 不视为 known。全部被手算用例或往返校验抓住。

## 范围与来源

上游 epfl-dlab/zip2zip-compression（2610.13605）：

- `src/codec.rs::encode_fn`（L213-303）：校验（ids 非空、`pad_token_id >= initial + max_codebook_size`、disabled 在基词表内且唯一、`max_subtokens >= 2`、truncation/padding 需 max_length）、主循环（disabled flush+单发、unknown 入本 `insert` 后 pop-emit-reset、`buffer.len() == max_subtokens` 强制发射、while 顶 truncation 早退）、pad 左填充、mask 按 pad 置 0。
- `src/codec.rs::decode_fn`（L310-385）：KwKwK 自指（未知 id 时 `buffer.push(buffer[0])` 后 `insert_buffer(id)` **用带进来的 id 入本**）、force-merge、码本满 `buffer.clear()` 冻结、while-merge 更新。本题只做 oracle，不出题。
- `src/config.rs`：`CodebookConfig` 各上限字段的默认与约束。

本题的口径差异：

- **不跑 tokenizer、不跑模型**：输入就是基词表 id 序列，超词 id 从 `initial_vocab_size` 起编号——与上游「超词挂在 tokenizer 词表尾部」的布局一致，但去掉了 tokenizer 接线。
- **solve 只做 encode**；decode 以独立 oracle 做往返校验（上游 decode 的 KwKwK 插入 quirk 适合当独立一题，不在本题出）。
- `padding="longest"`（批量级、按 batch 最长填充）属批量语义，不出；`attention_mask` 是编码产物，不做批量堆叠。
- 码本 dict 键序钉死为超词 id 升序（Rust HashMap 无序，Python 端补确定性契约）。
