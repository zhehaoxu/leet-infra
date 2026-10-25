# 评测：few-shot 提示的预算截断

## 题目描述

大模型评测框架（LM-eval-harness / OpenCompass 一类）在拼 few-shot 提示时，必须把演示条数压进模型的上下文预算里：演示**按固定顺序**一条一条往上添，一旦"再加一条就会超预算"就**当场停止**——注意是停止，不是跳过这条去试下一条更短的。同时，查询模板和**标签尾巴**（例如 `"Answer:"`，模型要在它后面续写）永远不能被裁掉，否则输出格式直接崩。Wang et al., *How does in-context learning work? A framework for understanding the successes and failures of LLMs*（arXiv:2201.11903）把这套"演示选择 + 预算截断"形式化成了示例选择算法。

为了摆脱对 HuggingFace tokenizer 的依赖（判题环境离线），本题自带一个**玩具分词器**（见代码模板）：按空白切词，词在 `VOCAB` 里就值 1 个 token，不在就展开成**逐字符** token（每个字符一个 `<unk>`）。

## 规则

- 输入：`demos`（`list[str]`，固定顺序的演示文本，可能为空）、`query`（`str`，查询模板）、`tail`（`str`，标签尾巴）、`budget`（`int`，整条 prompt 的 token 上限）。
- 组装顺序：被保留的演示（按给定顺序）→ `query` → `tail`，用单个空格连接。因为分词只认空白，**拼接不产生额外 token**，总长等于各部分 token 数之和。
- 预算从 `query` 与 `tail` 开始占：\(u_0 = \text{tok}(\text{query}) + \text{tok}(\text{tail})\)，它们**永不裁剪**——即使 \(u_0 > budget\) 也照样全部保留（如实报超支）。
- 贪心规则：对 \(i = 0, 1, 2, \dots\) 依次判断
  \[ u_i + \text{tok}(\text{demo}_i) \le budget \;\Rightarrow\; \text{收下第 } i \text{ 条},\ u_{i+1} = u_i + \text{tok}(\text{demo}_i) \]
  否则**立刻停止**，后面的演示一律不要（即使它们更短、即使换一条也装得下）。\(=\) 边界算"收下"。
- 返回二元组 `(kept, final_len)`：`kept` 是 int64 张量，形状 \((m,)\)，为被保留演示的下标（升序，且一定是从 0 开始的一段前缀）；`final_len` 是 Python int，等于组装后整条 prompt 的 token 数。
- 玩具分词器的定义与 `VOCAB` 写在代码模板里，必须照它的规则算长度；不许引入任何真实 tokenizer。

## 示例

`budget = 15`，`query = "Question:"`（1 token），`tail = "Answer:"`（1 token）：

```
demo0 = "the cat sat on the mat"   -> 6 token（全在词表里）
demo1 = "the dog is very nice"     -> 8 token（"very" 不在词表 -> 展开成 v,e,r,y 四个）
demo2 = "yes"                      -> 1 token
demo3 = "no"                       -> 1 token

u0 = 1 + 1 = 2
加 demo0：2 + 6 = 8  <= 15  -> 收下
加 demo1：8 + 8 = 16 >  15  -> 停止（demo2/demo3 虽各只 1 token，也不许再捡）

kept = [0]        final_len = 8
拼出来的 prompt = "the cat sat on the mat Question: Answer:"（正好 8 token）
```

## 约束

- CPU；纯整数运算，无随机性。
- \(0 \le budget\)，`demos` 可为空列表，单条演示可以短到只有空白（token 数 0）。
- **退化性质**：`demos` 为空 ⇒ `kept = []`、`final_len = tok(query) + tok(tail)`；`budget` 大于等于全部之和 ⇒ 全部保留；`budget < tok(query) + tok(tail)` ⇒ 一条演示都不放、`final_len` 如实超过 `budget`；恰好等于某次累加值 ⇒ 那条演示被保留。
- 输出 `kept` 必须是形如 `[0, 1, ..., m-1]` 的**前缀**，不能出现"跳过中间某条"的下标。
