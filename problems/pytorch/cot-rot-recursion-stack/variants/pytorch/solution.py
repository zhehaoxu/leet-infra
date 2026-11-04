"""RoT（Recursion of Thought, arXiv:2306.06891, soochan-lee/RoT）递归求解序列构造账。

官方把一道算术题按 rot 范式编译成一条 seq2seq 训练序列：子问题问答内联注入
（x 追加子问题 token、y 追加子问题 token + '<THINK>'，子答案只进 x、y 侧对应
位置补 '<PAD>'），尾递归（tail）子问题把首 token 换成 '<TAIL>'、不发射子答案、
主答案也不发射；ProbGraph 把题目递归展开成去重后的子问题图（DFS 先序）；
cot 范式则把思维链前序展平（问题 → 递归展开 → 答案）。

solve(mode=...) 四个入口（官方 data/tokenizer.py + data/problem.py
Problem.solve/_add_*_xyl/ProbGraph/_flatten_thought + data/arithmetic.py
Compare/Add/Mul，口径逐条钉死）：

    mode="tokenize"：text -> token 列表（'\\\\c' 转义把下一字符单切成 token、
        '<...>' 整体成一个特殊 token、未闭合的 '<...' 丢弃）。
    mode="rot"：problem + args -> (x, y, label)。problem ∈ {'compare','add',
        'mul'}；label: PAD=0 / Q=1 / T=2 / A=3。
    mode="flatten"：problem + args -> cot 范式思维链展平 token 列表
        （不去重，尾递归只出 '<TAIL>' 问题不出答案）。
    mode="graph"：roots（默认 [(problem, args)]）-> ProbGraph 展开序
        （DFS 先序 + 按 (problem, args) 节点去重，首现即展开）。
"""


LABEL_PAD, LABEL_Q, LABEL_T, LABEL_A = 0, 1, 2, 3
PROBLEMS = ('compare', 'add', 'mul')


def tokenize(s):
    """官方 data/tokenizer.py：转义/特殊 token 感知的字符切分。"""
    state = 'NORM'
    tokens = []
    token = ''
    for c in s:
        if state == 'NORM' and c == '\\':
            state = 'ESCAPE'
        elif state == 'ESCAPE':
            tokens.append(c)
            state = 'NORM'
        elif state == 'SPECIAL':
            if c == '>':
                tokens.append(token + c)
                state = 'NORM'
            else:
                token += c
        elif c == '<':
            state = 'SPECIAL'
            token = c
        else:
            tokens.append(c)
    return tokens


def _check_problem(problem):
    if problem not in PROBLEMS:
        raise ValueError(f"Unsupported problem {problem!r}")


def _check_args(args):
    if not isinstance(args, (tuple, list)) or len(args) != 2:
        raise ValueError("args must be a (left, right) pair")
    for v in args:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("args must be a (left, right) int pair")


def _question(problem, args):
    left, right = args
    if problem == 'compare':
        return f'<GO>{left}<VS>{right}='
    if problem == 'add':
        return f'<GO>{left}+{right}='
    return f'<GO>{left}*{right}='


def _answer(problem, args):
    left, right = args
    if problem == 'compare':
        if left > right:
            return '<GT><STOP>'
        if left < right:
            return '<LT><STOP>'
        return '<EQ><STOP>'
    if problem == 'add':
        return f'{left + right}<STOP>'
    return f'{left * right}<STOP>'


def _thought(problem, args):
    """官方 Compare/Add/Mul.thought：返回 [(sub_problem, sub_args[, 'tail'])]。"""
    if problem == 'compare':
        left, right = args
        if left < 10 and right < 10:
            return []
        thoughts = []
        if len(str(left)) == len(str(right)):
            l_first, r_first = int(str(left)[0]), int(str(right)[0])
            thoughts.append(('compare', (l_first, r_first)))
            if l_first == r_first:
                # 官方走 int(str(x)[1:])：前导零丢失（105 的 rest 是 5）
                thoughts.append(('compare',
                                 (int(str(left)[1:]), int(str(right)[1:]))))
        return thoughts
    if problem == 'add':
        left, right = args
        if left < 10 and right < 10:
            return []
        l_last, r_last = left % 10, right % 10
        thoughts = [('add', (l_last, r_last))]
        l_rest, r_rest = left // 10, right // 10
        if l_last + r_last >= 10:
            thoughts.append(('add', (l_rest, 1)))
            l_rest += 1
        if l_rest > 0 and r_rest > 0:
            thoughts.append(('add', (l_rest, r_rest)))
        return thoughts
    # mul
    left, right = args
    if left <= 1 or right <= 1:
        return []
    if left <= 9 and right <= 9:
        return []
    if right < 10:
        a1 = (left % 10) * right
        a2 = (left // 10) * right
        return [('mul', (left % 10, right)), ('mul', (left // 10, right)),
                ('add', (a2 * 10, a1), 'tail')]
    a1 = left * (right % 10)
    a2 = left * (right // 10)
    return [('mul', (left, right % 10)), ('mul', (left, right // 10)),
            ('add', (a2 * 10, a1), 'tail')]


def _add_thought_xyl(t_q, t_a, x, y, label):
    """官方 Problem._add_thought_xyl：x 收问题（+答案），y 收问题+<THINK>（答案侧补 PAD）。"""
    t_q = list(t_q)
    if t_a is None:
        t_q[0] = '<TAIL>'
    x += t_q
    y += t_q + ['<THINK>']
    label += [LABEL_T] * (len(t_q) + 1)
    if t_a is not None:
        x += t_a
        y += ['<PAD>'] * (len(t_a) - 1)
        label += [LABEL_PAD] * (len(t_a) - 1)


def rot_solve(problem, args):
    """官方 Problem.solve(paradigm='rot')：题目 -> (x, y, label) 训练序列。"""
    _check_problem(problem)
    _check_args(args)
    x = tokenize(_question(problem, args))
    y = x[1:]
    label = [LABEL_Q] * len(y)
    tail_recursion = False
    for thought in _thought(problem, args):
        sub_problem, sub_args = thought[0], thought[1]
        t_type = thought[2] if len(thought) > 2 else ''
        t_q = tokenize(_question(sub_problem, sub_args))
        if t_type == 'tail':
            tail_recursion = True
            t_a = None
        else:
            if tail_recursion:
                raise ValueError('Tail thought is not at the end')
            t_a = tokenize(_answer(sub_problem, sub_args))
        _add_thought_xyl(t_q, t_a, x, y, label)
    if not tail_recursion:
        answer = tokenize(_answer(problem, args))
        x += answer[:-1]
        y += answer
        label += [LABEL_A] * len(answer)
    return x, y, label


def _flatten(problem, args):
    flat = []
    for thought in _thought(problem, args):
        sub_problem, sub_args = thought[0], thought[1]
        t_type = thought[2] if len(thought) > 2 else ''
        q = tokenize(_question(sub_problem, sub_args))
        if t_type == 'tail':
            q[0] = '<TAIL>'
        flat += q
        flat += _flatten(sub_problem, sub_args)
        if t_type != 'tail':
            flat += tokenize(_answer(sub_problem, sub_args))
    return flat


def prob_graph(roots):
    """官方 ProbGraph.extend：DFS 先序展开 + 按 (problem, args) 节点去重。"""
    if not isinstance(roots, (list, tuple)):
        raise ValueError("roots must be a list of (problem, args)")
    order = []
    seen = set()

    def extend(probs):
        for prob in probs:
            if not isinstance(prob, (tuple, list)) or len(prob) != 2:
                raise ValueError("roots must be a list of (problem, args)")
            if prob in seen:
                continue
            seen.add(prob)
            order.append((prob[0], prob[1]))
            subprobs = [(t[0], t[1]) for t in _thought(prob[0], prob[1])]
            if subprobs:
                extend(subprobs)

    extend(list(roots))
    return order


def solve(mode="rot", problem=None, args=None, roots=None, text=None):
    """统一入口（判题协议）：按 mode 分发到 tokenize / rot / flatten / graph。"""
    if mode == 'tokenize':
        if not isinstance(text, str):
            raise ValueError("mode='tokenize' requires text: str")
        return tokenize(text)
    if mode == 'rot':
        if problem is None or args is None:
            raise ValueError("mode='rot' requires problem and args")
        return rot_solve(problem, args)
    if mode == 'flatten':
        if problem is None or args is None:
            raise ValueError("mode='flatten' requires problem and args")
        _check_problem(problem)
        _check_args(args)
        return _flatten(problem, args)
    if mode == 'graph':
        if roots is None:
            if problem is None or args is None:
                raise ValueError("mode='graph' requires roots or problem+args")
            roots = [(problem, args)]
        return prob_graph(roots)
    raise ValueError(f"Unsupported mode {mode!r}")
