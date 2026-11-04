"""RoT（Recursion of Thought）递归求解序列构造账 -- 你的实现。

官方（soochan-lee/RoT, arXiv:2306.06891）把一道算术题按 rot 范式编译成一条
seq2seq 训练序列；本题覆盖四个确定性账本（data/tokenizer.py + data/problem.py
Problem.solve/_add_*_xyl/ProbGraph/_flatten_thought + data/arithmetic.py
Compare/Add/Mul）。请实现下面的 solve；不允许 import torch / torchtext。

参数
----
mode : str
    "tokenize" | "rot" | "flatten" | "graph"（默认 "rot"）。
problem : str | None
    题目类名，∈ {'compare', 'add', 'mul'}（mode='rot'/'flatten' 必填；
    mode='graph' 与 args 搭配可省 roots）。
args : tuple[int, int] | None
    (left, right) 操作数对。
roots : list[(problem, args)] | None
    mode='graph' 的展开根列表；缺省时取 [(problem, args)]。
text : str | None
    mode='tokenize' 的待切分字符串。

返回
----
mode="tokenize"：list[str] —— '\\\\c' 转义把下一字符单切成 token；'<...>'
    整体成一个特殊 token（遇 '>' 收口）；未闭合的 '<...' 整段丢弃。
mode="rot"：(x, y, label)。
    x = tokenizer(问题) 起（首 token '<GO>'），随后 rot 循环：每个思维
    thought = (sub_problem, sub_args[, 'tail']) 先 tokenize(sub 问题)：
      - 非 tail：x 追加子问题 token 与子答案 token；y 追加子问题 token +
        '<THINK>'，再补 len(子答案 token) - 1 个 '<PAD>'；label 追加
        len(子问题)+1 个 T(=2) 与 len(子答案)-1 个 PAD(=0)。
      - tail：子问题首 token 换成 '<TAIL>'（x/y 双侧），不发射子答案，
        且主答案不发射（tail 后再出现非 tail 视为非法，ValueError）。
    非 tail 收尾：x 追加主答案 token[:-1]，y 追加主答案 token，label 追加
    len(主答案 token) 个 A(=3)。label: PAD=0 / Q=1 / T=2 / A=3。
mode="flatten"：cot 范式思维链前序展平（子问题 → 递归展开 → 子答案）；
    不去重；tail 子问题只出 '<TAIL>' 问题（含递归展开）不出答案。
mode="graph"：ProbGraph 展开序 list[(problem, args)]——DFS 先序，按节点
    (problem, args) 去重，首现即递归展开其子问题。

异常
----
problem 不在 {'compare','add','mul'}、mode 未注册、args/roots/text 形状
不符、tail 之后出现非 tail thought -> ValueError。
"""


def solve(mode="rot", problem=None, args=None, roots=None, text=None):
    raise NotImplementedError("TODO: 实现 tokenize / rot_solve / flatten / prob_graph")
