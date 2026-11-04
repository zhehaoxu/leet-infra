# CoCoGen：代码↔图双向解析账（图串解析 + add_edge 代码发射 + 代码还原）

## 题意

CoCoGen（arXiv:2210.07128, 官方仓 `reasoning-machines/CoCoGen`，原 `madaan/cocogen` 已迁移）把常识图表示成 python 代码（`class ExplanationDAG` + `add_edge` 行），`src/converters/explagraphs/relation.py` 提供确定性双向转换。回源修正：todos 原描述「环检测、根/叶约束、边标签合法性」与官方不符——官方**没有**独立的图校验器，确定性核心是三方账（图串解析 / 代码生成 / 代码还原）+ GraphAlgos 的 root 账 + case 转换三函数。本题全部显式入参，考纯账本：

```text
mode="parse":    graph_str -> {"nodes", "from_to", "from_to_typed"}
mode="to_python": belief + argument + stance + graph_str (+prompt_part_only) -> 代码串
mode="to_graph": code_str -> {"belief", "argument", "stance", "graph"}
mode="roots":    src_dest 邻接表 -> 入度 0 节点列表
mode="case":     text + case("camel"|"snake"|"desnake") -> 转换结果
```

## 账本规则（官方逐条钉死）

`mode="parse"`（官方 `get_nodes` L81-113）：

```text
graph_str[1:-1] 剥掉首尾各一个字符（不论是否括号）
按 ')(' 切分；不含 ';' 的段整段跳过
含 ';' 的段按 ';' 切分必须恰好 3 段（多/少 -> 解包 ValueError）；三段各自 strip
nodes 按首现序记录（先 from 后 to）
from_to / from_to_typed 只在 from 首现时初始化
to 已在 from_to_dict[from] 时两个账本都不追加
  （同 from→to 的第二条边——哪怕关系不同——整条静默丢弃）
```

`mode="to_python"`（官方 `graph_to_python` L10-79，relation 变体）：固定骨架
`class ExplanationDAG:` + 空行 + `__init__` + `belief =`/`argument =`/`stance =`
三行（**原样插值不转义**）+ 空行 + `# create a DAG to {stance} belief using argument`
注释；`prompt_part_only=True` 到此返回。否则解析图串，root 节点（入度 0、按
节点添加序）写 `begin = ["root1", "root2"]` 行（**带引号**），再按
`from_to_typed` 的 dict 序逐边发射 `add_edge("from", "rel", "to")` 行；结尾多一个空行。

`mode="to_graph"`（官方 `python_to_graph` L118-178，relation 变体）：正则
`belief = "(.*?)"` 等抽三字段（缺失 AttributeError 传播）；逐行含 `add_edge`
的行：`split("(")[1]` 取**第一个左括号之后**的内容（target 内的 `(` 会让后续
内容整体丢失）、`[:-1]` 砍掉最后一个字符，再按 `", "`+引号（逗号+空格+引号）
切分必须 3 段（越界 IndexError 传播），三段 strip 后拼 `(source; rel; target)`。
全部行拼完后统一清理：`"` 全删、`;` -> `; `、`  ` -> ` `（**单趟** replace，
三空格只折叠成两空格）。

`mode="roots"`（官方 `GraphAlgos.root_nodes` over `to_nx_graph`）：入度 0、
按节点添加序（`add_edge(src, dest)` 先加 src 后加 dest）；**空 dests 的 src
不入图**；纯环图无 root 返回 `[]`。

`mode="case"`（官方 `src/converters/utils.py`）：camel = 尾句号剥除 +
`split(' ')` + 逐 token `capitalize()` 连接（**其余字符转小写**）；snake =
尾句号剥除 + `re.sub(r'[\s\W]', '_', ...)`（空格与每个非字母数字字符各换
一个下划线）+ `lower()` + `strip()`（只剥空白，句号变出的尾 `_` 保留）；
desnake = `split('_')` 后 `' '.join`。空串在 camel/snake 的 `title[-1]` 处
IndexError（官方不捕获）。

## 关键怪癖（官方口径）

- parse 首尾剥**字符**不是剥括号：`X(a; rel; b)Y` 会解析出节点 `(a`、`b)`；
- 同 from→to 不同关系第二条边**整条丢弃**（dedup 检查同时门控无类型与带类型账本）；
- to_python 的 belief/argument/stance 不转义引号（含 `"` 的 belief 产生坏代码，官方忠实行为）；
- to_graph 的行解析靠 `split("(")[1]` + `[:-1]`：行内第二个 `(` 之后的内容丢失、行尾无 `)` 时吃掉末字符；双空格使 `", "` 分隔失配直接 IndexError；
- 清理三连是 `"` 删 -> `;` 加空格 -> 双空格折叠，**顺序与单趟**语义（`';  '` 不变 `'; '`、三空格不归一）；
- snake 的 strip 只剥空白：`Do A (Thing)!` -> `do_a__thing__`（尾 `_` 保留）。

## 输入约定

`graph_str` 是 `(from; rel; to)(...)` 串；`src_dest` 是 `{from: [to...]}` 邻接表；
case 的 snake/desnake 不是互逆（下划线无法区分来源）。
