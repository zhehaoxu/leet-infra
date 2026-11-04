# Moonshine 上下文偏置 Trie 走账

Moonshine（moonshine-ai/moonshine）的 `ContextBiaser`（`core/context-biaser.cpp`）在解码期把贪心 argmax 往 key term（术语/产品名/专有名词）上推——**零重训的 logits 加成账**：key term 分词后存进前缀 Trie，解码时追踪已走到的节点，argmax 前给「能延续某条活跃路径的 token」加 bonus。官方 doctest（`context-biaser-test.cpp`）把每个行为都钉了值，本题全部对拍官方数字。

## 机制拆解

### 1. Trie 注册（build）

- root = 节点 0；key term 的 token 序列逐个挂孩子；**共享前缀复用节点**（`Kubernetes` = `|Ku`+`bern`+`etes`，多词共前缀不重复建）；
- 重复注册同一序列：`sequence_count` +1 但**不加节点**；
- 空序列不注册——空 biaser 是精确 no-op。

### 2. 变体规则（variants）

tokenizer 用 U+2581（▁）标词首，句中与句首的同词首 token 不同，所以两个变体都要进 Trie：trim 后空 → `[]`；已带 ▁ 锚 → `[term]`（调用方说了算）；否则 `[term, " " + term]`。

### 3. 深度对数爬坡（bonus_for_depth）

`bonus = boost × (1 + ln depth)`（depth ≥ 1，默认 boost=2.0）。**完成比进入更值钱**：greedy 解码发出去的 token 收不回来，平坦 bonus 会让 key term 的短前缀在无关音频上乱触发；对数爬坡让「起头便宜、走完重赏」。depth 0 → 0。

### 4. apply：提案 + 去重取 max

对每个活跃节点，其全部孩子以 `bonus_for_depth(节点深度+1)` 入围；越界 token（<0 或 ≥vocab）跳过；**重叠 key term 提案同一 token 时取较大 bonus，不叠加**（官方注释：一个 token 被多条路径共享不能被加成到失衡）。空 biaser（sequence_count==0）直接 no-op。返回 `{token: bonus}`。

### 5. advance：root 常驻的走账

`next_active = [root] + 各活跃节点匹配 token 的孩子`：

- **root 永远活跃**——一个 term 可以在另一个 term 匹配到一半时重新起头（`[10,11]` 走到一半，`[11,20]` 仍可从 11 起步）;
- 不匹配的 token 让走账**回落到 root**（弃路即重启，没有部分匹配残留）。

### 6. 解码循环（bias_walk）

每步次序：`apply()` 的 bonus 落在 argmax **之前** → 选 token → `advance(选中的 token)`。reset 回 `[0]`（每次 decode 从 BOS 重来）。投机验证时 walk 跟**teacher-forced 前缀**而非预测（位置 t 条件的是前缀本身）——这是官方注释里明示的坑。

## 判题域

单入口 `solve(mode, **kw)`，七个 mode：`variants` / `build` / `bonus_for_depth` / `apply` / `advance` / `bonus_for_token` / `bias_walk`。纯 Python + math（无 torch RNG），数值全为官方 doctest 钉死的小数（boost=2 → depth1=2.0、depth2=2(1+ln2)、depth3=2(1+ln3)）。分词本身不出（token 序列显式入参）；投机验证分歧点账在流式状态题里另立。
