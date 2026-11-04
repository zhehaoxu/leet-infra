# 解题笔记

## 思路

官方 soochan-lee/RoT 的确定性核心：

1. **data/problem.py**：`Problem.solve(paradigm='rot')`（L104-135）——子问题问答内联注入 + tail 递归语义；`_add_thought_xyl` / `_add_answer_xyl` / `_init_question_xyl`（L137-165）——x/y/label 三流的对齐账；`ProbGraph.extend`（L310-319）——子问题图 DFS 先序去重展开；`_flatten_thought`（L369-380）——cot 范式前序展平；
2. **data/arithmetic.py**：`Compare.thought`（L48-68）、`Add.thought`（L114-133）、`Mul.thought`（L248-274）——三个 thought 递归（进位链、乘法分解 + tail 加法、首位/余位比较）；
3. **data/tokenizer.py**（L1-22）：转义/特殊 token 状态机。

回源修正：todos 原候选描述「GOAL/ANSWER 上下文栈账（压栈/弹栈、栈深上限、多上下文注入）」是**论文推理框架**的说法，官方仓实现的是 **seq2seq 训练序列构造**——子问题不压栈，而是连问答带 `<THINK>`/`<PAD>` 内联编译进一条序列；「栈」的对应物是 `ProbGraph` 的递归展开。

上游不入判题域：torchtext vocab（`build_vocab_from_iterator`）、`pad_sequence`/DataLoader、LSTM/Transformer 模型与训练、`generate()` 随机采样（操作数显式入参）、`ProblemSet.Iterator` 的 ammo/magazine 采样调度、gpt_eval 与 episodes yaml。

与现有题划界：
- **775/776/777**：demo 拼装、选题、候选验证；本题是**递归题目编译**——CoT 不是采样出来的文本而是从题目语法结构确定性生成；
- **rs-toT / rs-lats**：树搜索在推理时展开；本题在训练数据构造侧，且子问题由进位/分解规则机械派生；
- **data-tokenization 合集**：分词器题考 BPE/词表；本题 tokenizer 只是 22 行状态机（转义 + 特殊 token），核心在序列对齐账。

## 关键实现点（官方口径逐条钉死）

- **三流长度不变量**：`len(x)==len(y)==len(label)` 恒成立——init 时 y 比 x 少 1；每个 thought 给 y 多加 `'<THINK>'` 一个（x 只加 t_q）；非 tail 的子答案给 x 加 len(t_a)、y 加 len(t_a)-1 个 `<PAD>`（补回 1）；主答案 x 加 len-1、y 加 len（再补 1）；tail 序列没有主答案，以 `y[-1]=='<THINK>'` 收尾。这条不变量是 label_invariants 用例的骨架。
- **tail 递归语义**（`T(Add, (a2*10, a1), 'tail')`）：`t_q[0]` 换成 `'<TAIL>'`（官方直接下标替换，默认问题以 `<GO>` 开头）；t_a=None → 不发射子答案；`tail_recursion=True` → 主答案也不发射；官方用 `assert not tail_recursion` 拦「tail 后出现非 tail」，本题以 ValueError 等价实现（消息钉死 'Tail thought is not at the end'）。
- **Compare 的余位陷阱**：`int(str(l)[1:])` 对 105 产出 5（`'05'`→int 丢前导零）——(105,105) 的第二个 thought 是 `(5,5)` 而非 `(05,...)`；位数不同时零 thought（官方 docstring L14 的 73847<VS>7243 示例是陈旧注释，代码里 `if digit_l == digit_r` 拦住）。
- **Add 的进位链**：`l_last + r_last >= 10` 时追加 `(l_rest, 1)` 并**先追加后自增** `l_rest += 1`——(9,91) 的 l_rest=0 时进位 thought `(0,1)` 仍发生（先追加），随后 `(1,9)`；`(999,1)` 链 `(9,1)→(99,1)`（r_rest=0 拦住第三个）。
- **Mul 的双 base 与 tail 参数**：`l<=1 or r<=1`（×0/×1 记忆）先于 `l<=9 and r<=9`（九九表）；tail thought 的参数是 `(a2*10, a1)`——**a2 在高位**；`right=10` 走 `>=10` 支且 `r%10=0` 的部分积为 0（(12,10) 用例钉住）。
- **tokenizer 状态机**：`'\\c'` 吞反斜杠、单切下一字符（`'\\<x>'`→`['<','x','>']`，`'\\\\<x>'`→`['\\','<x>']`）；`'<...>'` 遇 `>` 收口（含空 `<>`）；循环结束处于 SPECIAL 时 token 缓冲**整段丢弃**。
- **ProbGraph 去重即剪枝**：`if prob in self: continue`——重复节点（Mul(11,11) 的两个 `Mul(11,1)`）只在首现处展开子树；返回序是 dict 插入序 = DFS 先序。

## 测试设计（11 例）

- `tokenize`：特殊 token/转义/未闭合/空 `<>`/`'\\<x>'` 与 `'\\\\<x>'` 双边界；期望值全部由官方仓实测钉死；
- `compare_paths`：base / 同位递归（105 前导零陷阱）/ 位数不等零 thought / 首位分胜负单 thought——全序列 x/y/label 断言；
- `add_no_carry`：(31,28) 全序列 + (5,123) 的 l_rest=0 拦截；
- `add_carry`：(39,28) 进位链、(999,1) 深链、(9,91) 的 `(0,1)` 进位 thought——全序列；
- `mul_tail`：(12,3)/(123,45)/(11,11 重复 thought 不去重)——`<TAIL>` 双侧替换、无 A label、y 以 `<THINK>` 收尾；
- `mul_base`：(3,4)/(37284,1)/(9,9 九九表边界)/(12,10 的 r=10 分解支)；
- `flatten_cot`：前序展平 + tail 只出问题 + 不去重（11,11 双份）；
- `graph_order`：15 节点 DFS 先序 / 多根按序 / 重复根去重 / 子树去重 / roots 缺省单根；
- `label_invariants`：9 组 (n_thoughts, n_pads, has_tail, n_ans) 账目核对——len 三流相等、`x.count('<THINK>')==0`、`y.count('<PAD>')`、tail 无 A 且 `'='`/`'<THINK>'` 收尾；
- `purity`；`args_validation`：mode/problem/args/roots/text 七类形状与类型 + bool 排除 + graph 的 tuple 根合法路径。

## 变异收口

41 条设计（含 1 组两段替换的 BFS 变异）→ **38 条 CAUGHT + 3 条等价剔除**：

- m-cmp-base-le9：`left < 10` ⟺ `left <= 9`（int 域恒等，`_check_args` 已保证 int）；
- m-args-len：形状检查删除后 `(1,)`/`(1,2,3)`/`"12"` 在 `_question` 解包或逐字符 int 检查处仍抛 ValueError；
- m-val-rot-req：mode='rot' 缺参兜底 raise 删除后 None 落 `_check_problem` 仍抛 ValueError（消息不同类型相同；m-graph-default-raise 同构未单列）。

覆盖：tokenizer 五态（escape 丢失/粘滞、特殊 token 丢 `>` 回显/丢 `<` 起始/未闭合保留）、对齐六态（y 不移位/`<THINK>` 丢失/T 计数/`<PAD>` y 侧与 label 侧/答案 x 侧不截/label 少 1/y 截断）、tail 三态（token 不换/子答案发射/主答案发射）、Compare 四态（base 边界/位数守卫/余位条件/比较符）、Add 五态（进位边界/自增丢失/双守卫各删/base 半删）、Mul 五态（×1 基例/九九表边界/分解支边界/tail 参数交换/分解序交换）、flatten 三态（前序颠倒/tail 出答案/tail token 不换）、graph 四态（去重删除→死递归/DFS→BFS/递归剪断/顶层形状检查删除）、校验六态。

## 工程学备注

- **官方 docstring 会撒谎**：Compare 的 docstring 示例 `73847<VS>7243` 展示了位数不等时的递归，但代码 `if digit_l == digit_r` 并不递归——一切以代码为准，docstring 里 7243 疑为 73413 的笔误。
- **先跑官方代码再写期望值**：全部 fixture 的 x/y/label 由 stub torchtext 后的官方仓直接 dump（/tmp/rotcheck），避免手算 32 元素 label 列表出错；(9,91) 的 `(0,1)` thought 与 (12,10) 的零部分积都是 dump 后才确认的。
- **变异脚本多段替换**：DFS→BFS 变异需要同时改 `def extend` 头与子树递归体——用两段 replace 合成一个变异（PAIR_MUTS 机制），单段替换会让 BFS 变异语法错误变成 fatal 误判。
- **判题协议**：判题器只注入单个 `solve`，四入口用 `mode` kwarg 分发（tokenize/rot/flatten/graph）。
