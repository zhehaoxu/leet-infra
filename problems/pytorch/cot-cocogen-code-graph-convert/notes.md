# 解题笔记

## 思路

官方 reasoning-machines/CoCoGen 的确定性核心（回源修正：todos 原描述「环检测、根/叶约束、边标签合法性」不成立——仓内 grep 无 is_valid/validate/cycle checker，GraphAlgos 只有 topo_sort/root_nodes/to_nx_graph/get_predecessors/get_successors 五个薄函数）：

1. **relation.py ExplagraphPythonConverterRelation.get_nodes**（L81-113）：`(from; rel; to)(...)` 图串解析账——首尾剥字符、`)(" 切分、无 `;` 段跳过、3 段解包、strip、首现去重、**to 重复整条丢弃**；
2. **graph_to_python**（L10-79，relation 变体）：固定骨架 + `begin = [...]`（root 账）+ 逐边 `add_edge("from", "rel", "to")` 发射；
3. **python_to_graph**（L118-178，relation 变体）：正则抽三字段 + 逐行 `split("(")[1][:-1]` + `", "`+引号切分 + 三段 strip + 清理三连（`"` 删 / `;` 加空格 / 双空格折叠）；
4. **GraphAlgos.root_nodes over to_nx_graph**（utils/algo_utils.py）：入度 0 按添加序；
5. **src/converters/utils.py 三函数**：to_camel_case（尾句号 + capitalize 连接）、to_snake_case（尾句号 + `[\s\W]`→`_` + lower + strip）、from_snake_to_normal_str（`_` 切分回空格）。

上游不入判题域：OpenAI 调用与 prompt 组装（make_codex_prompt 等）、tree/text/literal 等 converter 变体（机制同构，取 relation 版）、proscript/propara/boolgen 三族（流程脚本/属性转换，另一论文任务）、networkx 本体（root 账自实现）、compile_code_get_object 的 exec 兜底（syntax error 重试截尾，非本题账）、struct_eval/graph_metrics 的 GED/同构/JSD 指标（图对比，无解析语义）。

与现有题划界：
- **775-780**：demo 拼装/候选评分/后向验证/RoT 序列编译/CR 状态图/STaR 评测写回；本题是**表示层双向转换账**——输入输出都是纯文本，无控制流状态机；
- **776 的结构化解析**：候选解析后进入评分比较；本题解析即终点（round-trip 恒等是机制本体）。

## 关键实现点（官方口径逐条钉死）

- **parse 的首尾剥除是字符级**：`graph_str[1:-1]`——空串/单字符都安全（切片短于范围得空），`"(a)"` 剥完剩 `a` 无 `;` 全跳过；
- **3 段解包**：`from_node, edge_type, to_node = edge.split(";")`——段内 `;` 数 != 2 抛 ValueError（官方无捕获，传播）；`"(noise)"` 无 `;` 整段跳过；
- **to 重复的整条丢弃**：`if to_node not in from_to_dict[from_node]:` 同时门控 `from_to_dict.append` 与 `from_to_typed.append`——`(a; r1; b)(a; r2; b)` 的 r2 边不进任何账本；
- **to_python 骨架逐行**：class 行带**两个** `\n`、三字段原样插值（含 `"` 不转义）、注释行含 stance 插值、root 带引号 `", ".join(f'"{n}"')`、每行 add_edge 后统一结尾 `py_source + "\n"`；
- **root 账**：`to_nx_graph` 只 `add_edge`——**空 dests 的 src 不入图**（`{"b": []}` 的 roots 是 `[]` 而非 `["b"]`）；`add_edge(src, dest)` 先加 src 后加 dest 决定遍历序；
- **to_graph 行解析**：`line.split("(")[1]` 在 target 含 `(` 时截断其后所有内容（`add_edge("a", "r", "b (x)")` 得 `(a; r; b)`）；`[:-1]` 砍尾一字符——行尾无 `)` 时吃掉末引号仍可解析；`, "`+引号分隔在双空格时失配 → fields 只有 1 段 → `fields[1]` IndexError；
- **清理三连顺序**：`"` 删（行内残留引号统一清理）→ `;` -> `; `（把 junk 段的 `;` 也加空格）→ `"  "` -> `" "` **单趟**（三空格只折叠一层）；
- **snake 的字符账**：`re.sub(r'[\s\W]', '_', ...)` 每个空白/非字母数字字符各换**一个** `_`（无 `+` 量词）——双空格得双 `_`；`strip()` 只剥空白，句号剥除后 sub 出的尾 `_` 保留；`capitalize()` 是首字母大写**其余小写**（`doA` -> `Doa`，与 `token[0].upper()+token[1:]` 不同）；
- **空串**：camel/snake 的 `title[-1]` 对空串抛 IndexError——校验层不拦，官方异常原样传播。

## 测试设计（12 例）

parse 2（基本图 + 怪癖 8 路：首尾字符剥、无 `;` 段、空与 `(a)`、同 to 两种丢弃、4 段 ValueError、from 带空白 strip）、to_python 3（docstring 例全文精确、prompt_only 截断、root 序与不转义插值）、to_graph 2（round-trip 恒等 + 怪癖 7 路：无 belief AttributeError、短行/双空格 IndexError、无 `)` 行、target 内 `(` 截断、junk 尾巴、干净 round-trip）、roots 1（双 root 序、纯环 []、空表、空 dests src 不入图）、case 3（camel 含 capitalize 转小写与空串 IndexError、snake 四变体、desnake）、args_validation 1（11 路非法入参）。全部期望值由官方源码 exec 沙箱（ast 抽类/函数段、stub GraphPythonConverter/List/Dict，networkx 3.7 直通官方 GraphAlgos）实测钉死，含 to_python→to_graph round-trip 恒等断言。

## 回源偏差记录

- 官方 `get_nodes` 返回 `(nodes, from_to_dict, from_to_dict_typed)` 三元组（typed 值是 tuple 列表）；本题 parse 返回 dict（typed 值转 tuple 列表保持可比）。
- 官方 `graph_to_python`/`python_to_graph` 是方法且内部 print 调试；本题剥 print、骨架逐行拼接语义不变。
- 官方 root 账走 `networkx.DiGraph`；本题自实现 add_edge 序 + has_parent 集（与 nx 节点添加序/入度语义逐条对拍）。
- `GraphAlgos.topo_sort`（一行 nx 转发）与 predecessors/successors 不入题（无分支语义增量）。
