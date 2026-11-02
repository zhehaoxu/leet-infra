# notes（ag-coa-relay-ledger）

## 回源

- 论文：Chain of Agents（arXiv 2406.02818）§3.1 公式 (1) `CU_i = LLM_{W_i}(I_W, CU_{i-1}, c_i, q)`、
  §3.2 公式 (2) `Response = LLM_M(I_M, CU_l, q)`、Algorithm 1 的伪码、附录 B 的 worker / manager
  prompt 模板。这是**协议**的唯一权威来源。
- 论文**没有**官方代码仓库（arXiv abs 页与正文都没有给出链接）。分块算法在论文里只写到
  「每块短于 k、共 ⌈n/k⌉ 块」，token/word 计量、是否 overlap、边界如何对齐都没说，所以分块
  口径改用公开实现 `rudrankriyam/Chain-of-Agents` main 的 `utils.py::split_into_chunks`
  （第 43-88 行）。
- 本题只搬结构与账：LLM 调用→`worker_out` / `manager_out` 两条入参，`k`/`max_cu`/`window`
  三个上限参数，其余全是分块与接力。

## 论文口径（逐条核实）

1. **manager 只吃最后一个 CU**：§3.2 原文「the manager agent consumes accumulated knowledge from
   last worker `CU_l`」，附录 B 的 manager prompt 也写「the source text is too long and has been
   summarized」。脚注 1 更进一步：把**所有** `CU_i` 喂给 manager 会因互相冲突而掉分。所以账本里
   `manager_cus` 恒为单元素列表，`manager_input_words` 与块数、原文长度都无关。
2. **`CU_0` 是空串**：Algorithm 1 的 `Initialize CU_0 ← empty string`。因此第一个 worker 的
   `cu_in_words` 必为 0，这个断言能直接区分「从 0 开始」和「从第一个生成开始」。
3. **只传上一个 CU**：worker i 的输入是 `I_W + CU_{i-1} + c_i + q`，不是全部历史。中间 CU 再长
   也只影响下一轮，不会累积到 manager。
4. **无相关信息时是「重写并保留」，不是清空**：§3.1 描述 W3 在本块找不到信息时「directly
   rewrites CU_2 by putting the correct answer as the first token of CU_3 without adding any
   unrelated information」——旧信息仍在。题面落成「空生成 → 沿用上一个 CU」。
5. **按序处理，默认左到右**：§5.5 明确 left-to-right 最优（并与 right-to-left / permutation
   对比）。逆序只改 worker 的处理顺序，块表仍按原文顺序编号，`cu_l` 也随之变成原文**第一块**
   的 CU。
6. **`k` 同时是「块上限」和「agent 窗口」**：Algorithm 1 要求每块短于 `k`，而一个 worker 的输入
   `CU_{i-1} + c_i + q + I_W` 也必须塞进 `k`。题面把两件事拆成 `k`（分块）与 `window`（可选校验，
   默认关闭），因为论文对「CU 超长怎么办」完全没有规定，强制校验会让普通用例直接抛错。

## 公开实现补的口径

7. **段落优先**：先按 `\n\n` 切段，再按词数往当前块里塞；纯空白段跳过。这样块边界尽量落在
   段落边界上，而不是任意切断句子。
8. **段内空白被归一**：块文本是 `" ".join(words)`，所以段内换行、连续空格、首尾空白全部消失，
   段落之间的 `\n\n` 分隔符也不保留——原文的排版信息在这一步就丢了。
9. **超窗段落硬切成整窗块，尾巴留作起点**：`while len(words) > k` 切出 `words[:k]`，剩下的
   `words[k:]` 成为 `pending`；注意条件是**严格大于**，所以剩余长度恒在 `1..k`，永远不会是 0。
10. **满块滞留**：只有 `len(pending) + len(words) > k` 才落盘。攒满 `k` 词的块不会立刻落地，
    要等到下一段到来（或循环结束）才 flush，块的内容与顺序都因此不同——但**块数**仍等价于
    「贪心装满」。
11. **CU 超长只截断**：论文只给了生成参数（「Maximum generation token is set to 2048 for
    gemini-ultra and 1024 for the rest」），没给 CU 上限，也没说超了怎么办。题面的 `max_cu`
    是明确的工程约定：按词截断。截断必然走分词，所以顺带归一了空白；**不截断时原样返回**，
    这条差异让「总是归一」的实现无处藏身。
12. **`answer` 原样返回**：公开实现直接 `return response.choices[0].message.content`，不 strip。

## 与论文冲突的地方

- 公开实现的 `ManagerAgent.synthesize` 收的是**全部** `worker_outputs`，用
  `"Worker {i+1}: {output}"` 逐条拼进去；这与论文公式 (2) 和脚注 1 相反。本题按论文实现，
  并把「manager 只看 CU_l」做成显式账本字段与检查点——防止照着公开实现写的人一路绿灯。
- 公开实现没有 CU 长度上限、没有窗口校验、没有逆序选项，这三项都是题面另行声明的约定，
  不是论文或代码的口径，已在「输入」里写明。

## 等价变异

1. **不跳过纯空白段**（`if not para.strip(): continue` → 删除）：空白段 `split()` 得到 `[]`，
   `len(pending) + 0 > k` 只在 `pending > k` 时成立，而 `pending` 恒 ≤ `k`，所以必走
   `extend([])` 分支，什么也不做。20 000 例随机源零差异。
2. **硬切条件 `>` → `>=`**：`>=` 只是把「恰好 k 词」的那一次也切出去、并让 `pending` 变空，
   切出的整窗块与结尾 flush 的块内容一致、顺序一致，后续段落的分组也一致。20 000 例零差异。
3. **`num_chunks` 用 `len(walk)` 而非 `len(chunks)`**：`walk` 是 `range(len(chunks))` 或其反转，
   长度恒等，不可观测。

## 变异统计

- 实质变异 39 条全 CAUGHT，覆盖：分块（`>` / `>=`、超窗硬切、尾巴丢弃、块文本拼接方式、
  `split()` 口径）、词序区间（`end` 偏移、`pos` 步长）、处理顺序（不反转、旋转）、CU 链
  （清空、`== 0` / `<= 1`、截断阈值、按字符截断、总是归一、CU 赋值时机、记录原始生成）、
  输入词数（漏 query、漏指令、按字符算）、manager（喂全部 CU、把原文也算进输入、取第一个
  CU 当 `cu_l`、漏指令词、窗口校验删除）、校验（`k`/`max_cu` 边界、非法 order、条数、
  空源、窗口 `>=`）、账本字段（截断/沿用标志、`chunk_idx`、`words` 按字符、`chunk_words`
  直接写 `k`、`cu_out_words` 用原始生成、`cu_in_words` 用更新后的 CU、`answer` strip）。
- 等价 3 条（见上），其中 2 条有 20 000 例随机实证 + 结构性证明，1 条是长度恒等。

## 工程教训

- **「论文没说」要显式声明，不要假装是官方口径**：本题有三处（`max_cu` 截断、`window`
  校验、逆序选项）论文完全没提。把它们写进「输入」并在 notes 里点名，比混进 quirk 列表里
  诚实得多；否则复核时会以为是从代码里读出来的。
- **公开实现与论文冲突时，冲突本身就是最好的考点**：manager 喂全部 CU 这条如果只是「知道」，
  很容易写错；做成 `manager_cus` 字段 + 「原文变长但 CU_l 不变 → manager 输入不变」的断言后，
  照抄公开实现的解法必挂。
- **窗口校验别和分块窗口混用**：最初只有 `k`，`k` 同时管分块和输入校验，结果「恰好 k 词的块」
  加上 1 个词的问题就必然溢出——普通用例全变成报错。拆成 `k` + 可选 `window` 后才可用。
- **等价变异要用穷举证，别靠读代码**：「跳过空白段」「硬切 `>=`」两条第一眼都像会改变行为；
  写 20 000 例随机对比后零差异，才敢记成等价。反过来，「不产空块」那条一开始也被判成等价，
  补一个「空源 + 给一条生成」的用例后立刻变成 CAUGHT——差别在于**有没有被更早的校验兜住**。
- **`_bad` 只认 `ValueError`**：把 `k` 的校验改成抛 `TypeError` 也能被 `_bad` 抓到（走
  `except Exception` 分支报「wrong error」），但把校验整个删掉会让 `while len(words) > 0` 进入
  死循环、直接把变异进程打死。写变异时要先想清楚「删掉这条校验之后控制流会不会停不下来」。
