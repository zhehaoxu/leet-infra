"""RoT 递归求解序列构造 -- 测试用例（期望值全部由官方仓 soochan-lee/RoT 实测钉死）."""


# ---------- mode="tokenize" ----------

def check_tokenize(solve):
    got = solve(mode="tokenize", text=r'<GO>1\<3<x>=')
    assert got == ['<GO>', '1', '<', '3', '<x>', '='], got
    got = solve(mode="tokenize", text='<GO>a<THINK>b<unclosed')
    assert got == ['<GO>', 'a', '<THINK>', 'b'], got
    got = solve(mode="tokenize", text='12+34=')
    assert got == ['1', '2', '+', '3', '4', '='], got
    got = solve(mode="tokenize", text='a\\<b')
    assert got == ['a', '<', 'b'], got
    got = solve(mode="tokenize", text='<>')
    assert got == ['<>'], got
    # escape 吞掉 '<'（后续字符回 NORM）；'\\\\' 吞掉一个反斜杠
    got = solve(mode="tokenize", text='\\<x>')
    assert got == ['<', 'x', '>'], got
    got = solve(mode="tokenize", text='\\\\<x>')
    assert got == ['\\', '<x>'], got


# ---------- mode="rot"：Compare ----------

def check_compare_paths(solve):
    x, y, label = solve(mode="rot", problem="compare", args=(7, 7))
    assert x == ['<GO>', '7', '<VS>', '7', '=', '<EQ>'], x
    assert y == ['7', '<VS>', '7', '=', '<EQ>', '<STOP>'], y
    assert label == [1, 1, 1, 1, 3, 3], label

    # 同位数递归：rest 走 int(str(x)[1:])，前导零丢失（105 -> 05 -> 5）
    x, y, label = solve(mode="rot", problem="compare", args=(105, 105))
    assert x == ['<GO>', '1', '0', '5', '<VS>', '1', '0', '5', '=',
                 '<GO>', '1', '<VS>', '1', '=', '<EQ>', '<STOP>',
                 '<GO>', '5', '<VS>', '5', '=', '<EQ>', '<STOP>', '<EQ>'], x
    assert y == ['1', '0', '5', '<VS>', '1', '0', '5', '=',
                 '<GO>', '1', '<VS>', '1', '=', '<THINK>', '<PAD>',
                 '<GO>', '5', '<VS>', '5', '=', '<THINK>', '<PAD>',
                 '<EQ>', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 0, 3, 3], label

    # 位数不同：无 thought，直接出主答案
    x, y, label = solve(mode="rot", problem="compare", args=(84, 2))
    assert x == ['<GO>', '8', '4', '<VS>', '2', '=', '<GT>'], x
    assert y == ['8', '4', '<VS>', '2', '=', '<GT>', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 3, 3], label

    # 同位数但首位已分胜负：只比首位
    x, y, label = solve(mode="rot", problem="compare", args=(10, 29))
    assert x == ['<GO>', '1', '0', '<VS>', '2', '9', '=',
                 '<GO>', '1', '<VS>', '2', '=', '<LT>', '<STOP>', '<LT>'], x
    assert y == ['1', '0', '<VS>', '2', '9', '=',
                 '<GO>', '1', '<VS>', '2', '=', '<THINK>', '<PAD>',
                 '<LT>', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 3, 3], label


# ---------- mode="rot"：Add ----------

def check_add_no_carry(solve):
    x, y, label = solve(mode="rot", problem="add", args=(31, 28))
    assert x == ['<GO>', '3', '1', '+', '2', '8', '=',
                 '<GO>', '1', '+', '8', '=', '9', '<STOP>',
                 '<GO>', '3', '+', '2', '=', '5', '<STOP>', '5', '9'], x
    assert y == ['3', '1', '+', '2', '8', '=',
                 '<GO>', '1', '+', '8', '=', '<THINK>', '<PAD>',
                 '<GO>', '3', '+', '2', '=', '<THINK>', '<PAD>',
                 '5', '9', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 0, 3, 3, 3], label

    # l_rest == 0（left 只有 1 位）→ 第三个 thought 不发生
    x, y, label = solve(mode="rot", problem="add", args=(5, 123))
    assert x == ['<GO>', '5', '+', '1', '2', '3', '=',
                 '<GO>', '5', '+', '3', '=', '8', '<STOP>', '1', '2', '8'], x
    assert y == ['5', '+', '1', '2', '3', '=',
                 '<GO>', '5', '+', '3', '=', '<THINK>', '<PAD>',
                 '1', '2', '8', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 3, 3, 3, 3], label


def check_add_carry(solve):
    x, y, label = solve(mode="rot", problem="add", args=(39, 28))
    assert x == ['<GO>', '3', '9', '+', '2', '8', '=',
                 '<GO>', '9', '+', '8', '=', '1', '7', '<STOP>',
                 '<GO>', '3', '+', '1', '=', '4', '<STOP>',
                 '<GO>', '4', '+', '2', '=', '6', '<STOP>', '6', '7'], x
    assert y == ['3', '9', '+', '2', '8', '=',
                 '<GO>', '9', '+', '8', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<GO>', '3', '+', '1', '=', '<THINK>', '<PAD>',
                 '<GO>', '4', '+', '2', '=', '<THINK>', '<PAD>',
                 '6', '7', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 0, 3, 3, 3], label

    x, y, label = solve(mode="rot", problem="add", args=(999, 1))
    assert x == ['<GO>', '9', '9', '9', '+', '1', '=',
                 '<GO>', '9', '+', '1', '=', '1', '0', '<STOP>',
                 '<GO>', '9', '9', '+', '1', '=', '1', '0', '0', '<STOP>',
                 '1', '0', '0', '0'], x
    assert y == ['9', '9', '9', '+', '1', '=',
                 '<GO>', '9', '+', '1', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<GO>', '9', '9', '+', '1', '=', '<THINK>',
                 '<PAD>', '<PAD>', '<PAD>', '1', '0', '0', '0', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 0, 0, 0,
                     3, 3, 3, 3, 3], label

    # 进位发生在 l_rest == 0 之前：Add(0,1) 也会成为 thought
    x, y, label = solve(mode="rot", problem="add", args=(9, 91))
    assert x == ['<GO>', '9', '+', '9', '1', '=',
                 '<GO>', '9', '+', '1', '=', '1', '0', '<STOP>',
                 '<GO>', '0', '+', '1', '=', '1', '<STOP>',
                 '<GO>', '1', '+', '9', '=', '1', '0', '<STOP>',
                 '1', '0', '0'], x
    assert y == ['9', '+', '9', '1', '=',
                 '<GO>', '9', '+', '1', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<GO>', '0', '+', '1', '=', '<THINK>', '<PAD>',
                 '<GO>', '1', '+', '9', '=', '<THINK>', '<PAD>', '<PAD>',
                 '1', '0', '0', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 0, 0,
                     3, 3, 3, 3], label


# ---------- mode="rot"：Mul（tail 递归）----------

def check_mul_tail(solve):
    x, y, label = solve(mode="rot", problem="mul", args=(12, 3))
    assert x == ['<GO>', '1', '2', '*', '3', '=',
                 '<GO>', '2', '*', '3', '=', '6', '<STOP>',
                 '<GO>', '1', '*', '3', '=', '3', '<STOP>',
                 '<TAIL>', '3', '0', '+', '6', '='], x
    assert y == ['1', '2', '*', '3', '=',
                 '<GO>', '2', '*', '3', '=', '<THINK>', '<PAD>',
                 '<GO>', '1', '*', '3', '=', '<THINK>', '<PAD>',
                 '<TAIL>', '3', '0', '+', '6', '=', '<THINK>'], y
    assert label == [1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 2], label

    x, y, label = solve(mode="rot", problem="mul", args=(123, 45))
    assert x == ['<GO>', '1', '2', '3', '*', '4', '5', '=',
                 '<GO>', '1', '2', '3', '*', '5', '=', '6', '1', '5', '<STOP>',
                 '<GO>', '1', '2', '3', '*', '4', '=', '4', '9', '2', '<STOP>',
                 '<TAIL>', '4', '9', '2', '0', '+', '6', '1', '5', '='], x
    assert y == ['1', '2', '3', '*', '4', '5', '=',
                 '<GO>', '1', '2', '3', '*', '5', '=',
                 '<THINK>', '<PAD>', '<PAD>', '<PAD>',
                 '<GO>', '1', '2', '3', '*', '4', '=',
                 '<THINK>', '<PAD>', '<PAD>', '<PAD>',
                 '<TAIL>', '4', '9', '2', '0', '+', '6', '1', '5', '=',
                 '<THINK>'], y
    assert label == [1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], label

    # 两个相同 thought（Mul(11,1) 出现两次）：rot 序列不去重
    x, y, label = solve(mode="rot", problem="mul", args=(11, 11))
    assert x == ['<GO>', '1', '1', '*', '1', '1', '=',
                 '<GO>', '1', '1', '*', '1', '=', '1', '1', '<STOP>',
                 '<GO>', '1', '1', '*', '1', '=', '1', '1', '<STOP>',
                 '<TAIL>', '1', '1', '0', '+', '1', '1', '='], x
    assert y == ['1', '1', '*', '1', '1', '=',
                 '<GO>', '1', '1', '*', '1', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<GO>', '1', '1', '*', '1', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<TAIL>', '1', '1', '0', '+', '1', '1', '=', '<THINK>'], y
    assert label == [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 2, 2], label


def check_mul_base(solve):
    x, y, label = solve(mode="rot", problem="mul", args=(3, 4))
    assert x == ['<GO>', '3', '*', '4', '=', '1', '2'], x
    assert y == ['3', '*', '4', '=', '1', '2', '<STOP>'], y
    assert label == [1, 1, 1, 1, 3, 3, 3], label

    # right == 1：left <= 1 or right <= 1 基例
    x, y, label = solve(mode="rot", problem="mul", args=(37284, 1))
    assert x == ['<GO>', '3', '7', '2', '8', '4', '*', '1', '=',
                 '3', '7', '2', '8', '4'], x
    assert y == ['3', '7', '2', '8', '4', '*', '1', '=',
                 '3', '7', '2', '8', '4', '<STOP>'], y
    assert label == [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3], label

    # 9*9：left <= 9 and right <= 9 基例（记牢九九表，无 thought）
    x, y, label = solve(mode="rot", problem="mul", args=(9, 9))
    assert x == ['<GO>', '9', '*', '9', '=', '8', '1'], x
    assert y == ['9', '*', '9', '=', '8', '1', '<STOP>'], y
    assert label == [1, 1, 1, 1, 3, 3, 3], label

    # right == 10：走 >=10 分解支（right % 10 == 0）
    x, y, label = solve(mode="rot", problem="mul", args=(12, 10))
    assert x == ['<GO>', '1', '2', '*', '1', '0', '=',
                 '<GO>', '1', '2', '*', '0', '=', '0', '<STOP>',
                 '<GO>', '1', '2', '*', '1', '=', '1', '2', '<STOP>',
                 '<TAIL>', '1', '2', '0', '+', '0', '='], x
    assert y == ['1', '2', '*', '1', '0', '=',
                 '<GO>', '1', '2', '*', '0', '=', '<THINK>', '<PAD>',
                 '<GO>', '1', '2', '*', '1', '=', '<THINK>', '<PAD>', '<PAD>',
                 '<TAIL>', '1', '2', '0', '+', '0', '=', '<THINK>'], y
    assert label == [1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 0,
                     2, 2, 2, 2, 2, 2, 2, 0, 0,
                     2, 2, 2, 2, 2, 2, 2, 2], label


# ---------- mode="flatten" ----------

def check_flatten_cot(solve):
    got = solve(mode="flatten", problem="mul", args=(12, 3))
    assert got == ['<GO>', '2', '*', '3', '=', '6', '<STOP>',
                   '<GO>', '1', '*', '3', '=', '3', '<STOP>',
                   '<TAIL>', '3', '0', '+', '6', '=',
                   '<GO>', '0', '+', '6', '=', '6', '<STOP>'], got

    # 前序展平不去重：Mul(11,1) 出现两次
    got = solve(mode="flatten", problem="mul", args=(11, 11))
    assert got == ['<GO>', '1', '1', '*', '1', '=', '1', '1', '<STOP>',
                   '<GO>', '1', '1', '*', '1', '=', '1', '1', '<STOP>',
                   '<TAIL>', '1', '1', '0', '+', '1', '1', '=',
                   '<GO>', '0', '+', '1', '=', '1', '<STOP>',
                   '<GO>', '1', '1', '+', '1', '=',
                   '<GO>', '1', '+', '1', '=', '2', '<STOP>', '1', '2', '<STOP>'], got

    got = solve(mode="flatten", problem="add", args=(31, 28))
    assert got == ['<GO>', '1', '+', '8', '=', '9', '<STOP>',
                   '<GO>', '3', '+', '2', '=', '5', '<STOP>'], got


# ---------- mode="graph" ----------

def check_graph_order(solve):
    got = solve(mode="graph", roots=[('mul', (12, 34))])
    assert got == [('mul', (12, 34)), ('mul', (12, 4)), ('mul', (2, 4)),
                   ('mul', (1, 4)), ('add', (40, 8)), ('add', (0, 8)),
                   ('mul', (12, 3)), ('mul', (2, 3)), ('mul', (1, 3)),
                   ('add', (30, 6)), ('add', (0, 6)),
                   ('add', (360, 48)), ('add', (36, 4)), ('add', (6, 4)),
                   ('add', (3, 1))], got

    # 多根按序展开
    got = solve(mode="graph", roots=[('add', (39, 28)), ('add', (31, 28))])
    assert got == [('add', (39, 28)), ('add', (9, 8)), ('add', (3, 1)),
                   ('add', (4, 2)), ('add', (31, 28)), ('add', (1, 8)),
                   ('add', (3, 2))], got

    # 重复根去重
    got = solve(mode="graph", roots=[('add', (39, 28)), ('add', (39, 28))])
    assert got == [('add', (39, 28)), ('add', (9, 8)), ('add', (3, 1)),
                   ('add', (4, 2))], got

    # Mul(11,11) 两个相同 thought 在图中去重，子树只展开一次
    got = solve(mode="graph", roots=[('mul', (11, 11))])
    assert got == [('mul', (11, 11)), ('mul', (11, 1)), ('add', (110, 11)),
                   ('add', (0, 1)), ('add', (11, 1)), ('add', (1, 1))], got

    # roots 缺省时退化为单根 [(problem, args)]
    got = solve(mode="graph", problem="add", args=(31, 28))
    assert got == [('add', (31, 28)), ('add', (1, 8)), ('add', (3, 2))], got


# ---------- label 不变量 ----------

def check_label_invariants(solve):
    # (problem, args, n_thoughts, n_pads, has_tail, n_answer_tokens)
    cases = [
        ('compare', (7, 7), 0, 0, False, 2),
        ('compare', (105, 105), 2, 2, False, 2),
        ('add', (31, 28), 2, 2, False, 3),
        ('add', (39, 28), 3, 4, False, 3),
        ('add', (9, 91), 3, 5, False, 4),
        ('mul', (12, 3), 3, 2, True, 0),
        ('mul', (12, 10), 3, 3, True, 0),
        ('mul', (123, 45), 3, 6, True, 0),
        ('mul', (3, 4), 0, 0, False, 3),
        ('mul', (9, 9), 0, 0, False, 3),
        ('mul', (37284, 1), 0, 0, False, 6),
    ]
    for problem, args, n_thoughts, n_pads, has_tail, n_ans in cases:
        x, y, label = solve(mode="rot", problem=problem, args=args)
        assert len(x) == len(y) == len(label), (problem, args, len(x), len(y), len(label))
        assert x[0] == '<GO>', (problem, args, x[0])
        assert y[0] == x[1], (problem, args, y[0])
        assert x.count('<THINK>') == 0, (problem, args)
        assert y.count('<THINK>') == n_thoughts, (problem, args, y.count('<THINK>'))
        assert y.count('<PAD>') == n_pads, (problem, args, y.count('<PAD>'))
        if has_tail:
            assert label.count(3) == 0, (problem, args)
            # 主答案缺席：x 以 tail 子问题的 '=' 收尾，y 以 '<THINK>' 收尾
            assert x[-1] == '=' and y[-1] == '<THINK>' and label[-1] == 2, (problem, args)
        else:
            assert y[-1] == '<STOP>' and label[-1] == 3, (problem, args)
            assert label.count(3) == n_ans, (problem, args, label.count(3))
            assert x[-1] != '<STOP>', (problem, args)


# ---------- 纯度 ----------

def check_purity(solve):
    args = (39, 28)
    roots = [('mul', (12, 34))]
    x1, y1, l1 = solve(mode="rot", problem="add", args=args)
    g1 = solve(mode="graph", roots=roots)
    x2, y2, l2 = solve(mode="rot", problem="add", args=args)
    g2 = solve(mode="graph", roots=roots)
    assert x1 == x2 and y1 == y2 and l1 == l2 and g1 == g2
    assert args == (39, 28) and roots == [('mul', (12, 34))]
    assert x1 is not x2 and g1 is not g2


# ---------- 参数校验 ----------

def check_args_validation(solve):
    for bad in (
        lambda: solve(mode="nope", problem="add", args=(1, 2)),
        lambda: solve(mode="rot", problem="div", args=(1, 2)),
        lambda: solve(mode="rot", args=(1, 2)),
        lambda: solve(mode="rot", problem="add"),
        lambda: solve(mode="rot", problem="add", args=(1,)),
        lambda: solve(mode="rot", problem="add", args=(1, 2, 3)),
        lambda: solve(mode="rot", problem="add", args="12"),
        lambda: solve(mode="rot", problem="add", args=(1, "2")),
        lambda: solve(mode="rot", problem="add", args=(1, True)),
        lambda: solve(mode="tokenize"),
        lambda: solve(mode="tokenize", text=5),
        lambda: solve(mode="graph", roots="nope"),
        lambda: solve(mode="graph", roots=5),
        lambda: solve(mode="graph", roots=[("add", (1,))]),
        lambda: solve(mode="graph"),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    # graph 允许 (problem, args) 形式的 tuple 根
    got = solve(mode="graph", roots=(('add', (31, 28)),))
    assert got == [('add', (31, 28)), ('add', (1, 8)), ('add', (3, 2))], got


check_tokenize.example = True
check_compare_paths.example = True
check_add_no_carry.example = True
check_add_carry.example = True
check_mul_tail.example = True
check_mul_base.example = True
check_flatten_cot.example = True
check_graph_order.example = True
check_label_invariants.example = True
check_purity.example = True
check_args_validation.example = True
