# RoT：递归求解序列构造

## 题意

RoT（Recursion of Thought, arXiv:2306.06891, 官方仓 `soochan-lee/RoT`）把一道算术题按 rot 范式**编译成一条 seq2seq 训练序列**：子问题问答内联注入到题目序列里，尾递归子问题替换主答案。本题要求按官方口径实现四个确定性账本（纯字符串/列表，无模型）：

```text
tokenize:   转义/特殊 token 感知的字符切分
rot:        题目 -> (x, y, label) 训练序列
flatten:    cot 范式思维链前序展平
graph:      ProbGraph 子问题图 DFS 先序去重展开
```

`mode="rot"` 输入 `problem ∈ {'compare','add','mul'}` 与操作数对 `(left, right)`，输出三元组；`label: PAD=0 / Q=1 / T=2 / A=3`。

## 账本规则（官方逐条钉死）

```text
x = tokenize(question)              # '<GO>...' 起
y = x[1:]                           # 右移一位（seq2seq teacher forcing 对齐）
label = [Q] * len(y)
对每个 thought = (sub, sub_args[, 'tail'])：
    t_q = tokenize(sub 的 question)
    非 tail：t_a = tokenize(sub 的 answer)
        x += t_q;  x += t_a
        y += t_q + ['<THINK>'];  y += ['<PAD>'] * (len(t_a) - 1)
        label += [T] * (len(t_q) + 1);  label += [PAD] * (len(t_a) - 1)
    tail：t_q[0] 换成 '<TAIL>'（x/y 双侧），无子答案，
        且主答案不发射；tail 之后再出现非 tail 视为非法
非 tail 收尾：
    x += answer[:-1];  y += answer;  label += [A] * len(answer)
```

三个题目类的 thought 递归（`args` 均为非负整数对）：

- **Compare**（`<GO>l<VS>r=`）：双 base（`l<10 and r<10` 直接出 `<GT>/<LT>/<EQ>`）；同位数才递归——比首位（`int(str(l)[0])`），首位相等再比余位（`int(str(l)[1:])`，**前导零丢失**：105 的余位是 5）；位数不同零 thought。
- **Add**（`<GO>l+r=`）：base `l<10 and r<10`；先比个位（`l%10, r%10`），`>=10` 进位则追加 `(l//10, 1)` 并 `l//10 += 1`；`l_rest>0 and r_rest>0` 才追加余位加法。
- **Mul**（`<GO>l*r=`）：base `l<=1 or r<=1`（乘 0/1 记忆）与 `l<=9 and r<=9`（九九表记忆）；`r<10` 分解为 `(l%10, r)`、`(l//10, r)`，否则 `(l, r%10)`、`(l, r//10)`；最后追加 **tail** thought `(a2*10, a1)`（a1/a2 为两次部分积），主答案即该加法的结果——由下一层算出。

tokenizer：`'\\c'` 转义把下一字符单切成 token（吞掉反斜杠）；`'<...>'` 遇 `>` 收口整体成 token（含 `<>` 空特殊 token）；未闭合的 `'<...'` 整段丢弃。

ProbGraph：`extend` 按节点 `(problem, args)` 去重（首现即递归展开其子树），返回 DFS 先序展开序；cot 范式 `_flatten_thought` 前序展平（子问题 → 递归展开 → 子答案，tail 只出 `'<TAIL>'` 问题不出答案）且**不去重**。

## 陷阱

- **y 侧比 x 侧多收一个 token**：每个 thought 给 y 加 `'<THINK>'` 而主答案只让 x 少收最后一个 token——不变量 `len(x)==len(y)==len(label)` 靠「base 时 y=x-1 → 每 thought 补 1 → 答案再补 1」维持，tail 序列则以 `y[-1]=='<THINK>'` 收尾。
- **`<TAIL>` 替换的是 x/y 双侧的子问题首 token**，且主答案与子答案都不发射（label 无 A）；但非 tail 子答案里的 `'<STOP>'` 仍在序列里。
- **`int(str(l)[1:])` 前导零丢失**（105 → 5）与 **`l_rest=0` 时进位 thought `(0,1)` 仍会发生**（9+91）都是官方真实行为。
- Mul 分解支的边界：`right=10` 走 `>=10` 支（`r%10=0` 的部分积为 0）。

## 判题环境

无 torch / torchtext 依赖——官方 `Problem.solve` 的 vocab/`pad_sequence` 外壳不入题，只考 token 串与 label 的纯账本；随机采样（`generate`）不入，操作数显式入参。
