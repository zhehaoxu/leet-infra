"""CoCoGen（arXiv:2210.07128, reasoning-machines/CoCoGen）代码↔图双向解析账。

官方把「常识图」表示成 python 代码（class ExplanationDAG + add_edge 行），
并在 src/converters/explagraphs/relation.py 提供确定性的双向转换：
get_nodes 把 `(from; rel; to)(...)` 图串解析成节点/邻接表，graph_to_python
把图串发射成 add_edge 代码（root 节点经 GraphAlgos.root_nodes 取入度 0
写入 begin 行），python_to_graph 把代码逐行解析回图串。本题把三方账
（解析、代码生成、代码解析）与 GraphAlgos root 账、case 转换三函数全部
显式入参，考纯账本。

solve(mode=...) 五个入口（官方逐条钉死）：

    mode="parse"（get_nodes）：graph_str -> {"nodes", "from_to",
        "from_to_typed"}。graph_str[1:-1] 剥掉**首尾各一个字符**（不论
        是否括号），按 ')(' 切分；不含 ';' 的段整段跳过；含 ';' 的段按
        ';' 切分**必须恰好 3 段**（否则解包 ValueError）；三段各自 strip。
        nodes 按首现序记录（先 from 后 to）；from_to/from_to_typed 只在
        from 首现时初始化；**to 已在 from_to_dict[from] 时两个账本都不
        追加**（同 from→to 的第二条边——哪怕关系不同——整条静默丢弃）。

    mode="to_python"（graph_to_python，relation 变体）：belief + argument
        + stance + graph_str (+prompt_part_only) -> 代码串。
        固定骨架：class ExplanationDAG / __init__ / belief= argument=
        stance= 三行（原样插值，不转义）/ 空行 / '# create a DAG to
        {stance} belief using argument' 注释；prompt_part_only=True 到此
        返回。否则解析图串，root 节点（入度 0，按节点添加序）写
        `begin = ["root1", "root2"]` 行，再按 from_to_typed 的 dict 序
        逐边发射 `add_edge("from", "rel", "to")` 行；结尾多一个空行。

    mode="to_graph"（python_to_graph，relation 变体）：code_str ->
        {"belief", "argument", "stance", "graph"}。先用正则
        `belief = "(.*?)"` 等抽三字段（缺失则 AttributeError 传播）；逐行
        含 'add_edge' 的行：split("(")[1] 取**第一个左括号之后**的内容、
        [:-1] 砍掉最后一个字符，再按 ', "'（逗号+空格+引号）切分**必须
        3 段**（fields[1]/fields[2] 越界 IndexError 传播），三段 strip 后
        拼 `(source; rel; target)`。全部行拼完后统一清理：'"' 全删、
        ';' -> '; '、'  ' -> ' '（**单趟** replace，三空格只折叠成两空格）。

    mode="roots"（GraphAlgos.root_nodes over to_nx_graph）：src_dest
        邻接表 -> 入度 0 的节点列表（按节点添加序：dict 序先 src 后各
        dest；有父的节点出局；纯环图无 root -> []）。

    mode="case"（src/converters/utils.py 三函数）：case="camel"|"snake"|
        "desnake"。
          camel：尾句号剥除 -> split(' ') -> 逐 token capitalize() 连接；
          snake：尾句号剥除 -> `re.sub(r'[\\s\\W]', '_', ...)`（空格与一切
              非字母数字字符各换**一个**下划线）-> lower() -> strip()
              （只剥空白，句号变出的尾 `_` 保留）；
          desnake：split('_') -> ' '.join。
        空串在 camel/snake 的 title[-1] 处 IndexError（官方不捕获）。
"""
import re

MODES = ("parse", "to_python", "to_graph", "roots", "case")
CASES = ("camel", "snake", "desnake")


def _check_str(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str")


def _check_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _get_nodes(graph_str):
    """官方 get_nodes：图串 -> (nodes, from_to, from_to_typed)。"""
    nodes = []
    from_to_dict = dict()
    from_to_dict_typed = dict()
    graph_str = graph_str[1:-1]
    graph_parts = graph_str.split(")(")
    for edge in graph_parts:
        if ";" in edge:
            from_node, edge_type, to_node = edge.split(";")
            from_node = from_node.strip()
            to_node = to_node.strip()
            edge_type = edge_type.strip()
            if from_node not in nodes:
                nodes.append(from_node)
            if to_node not in nodes:
                nodes.append(to_node)
            if from_node not in from_to_dict:
                from_to_dict[from_node] = []
                from_to_dict_typed[from_node] = []
            if to_node not in from_to_dict[from_node]:
                from_to_dict[from_node].append(to_node)
                from_to_dict_typed[from_node].append((edge_type, to_node))
    return nodes, from_to_dict, from_to_dict_typed


def _root_nodes(src_dest_dict):
    """官方 GraphAlgos.root_nodes over to_nx_graph：入度 0，按添加序。

    to_nx_graph 只做 add_edge(src, dest)（先加 src 后加 dest）——空 dests
    的 src 不入图。
    """
    order = []
    has_parent = set()

    def _add(node):
        if node not in order:
            order.append(node)

    for src, dests in src_dest_dict.items():
        for dest in dests:
            _add(src)
            _add(dest)
            has_parent.add(dest)
    return [node for node in order if node not in has_parent]


def _graph_to_python(belief, argument, stance, graph_str, prompt_part_only):
    """官方 graph_to_python（relation 变体）。"""
    py_source = ""
    py_source += "class ExplanationDAG:\n\n"
    py_source += "    def __init__(self):\n"
    py_source += f"        belief = \"{belief}\"\n"
    py_source += f"        argument = \"{argument}\"\n"
    py_source += f"        stance = \"{stance}\"\n"
    py_source += "\n"
    py_source += f"        # create a DAG to {stance} belief using argument\n"
    if prompt_part_only:
        return py_source
    nodes, from_to_dict, from_to_dict_typed = _get_nodes(graph_str)
    root_nodes = _root_nodes(from_to_dict)
    root_nodes_str = ", ".join([f"\"{node}\"" for node in root_nodes])
    py_source += f"        begin = [{root_nodes_str}]\n"
    for node in from_to_dict_typed:
        edges = from_to_dict_typed[node]
        for (edge_type, edge_to) in edges:
            py_source += f"        add_edge(\"{node}\", \"{edge_type}\", \"{edge_to}\")\n"
    return py_source + "\n"


def _python_to_graph(code_str):
    """官方 python_to_graph（relation 变体）。"""
    belief = re.search(r"belief = \"(.*?)\"", code_str).group(1)
    argument = re.search(r"argument = \"(.*?)\"", code_str).group(1)
    stance = re.search(r"stance = \"(.*?)\"", code_str).group(1)
    code_lines = code_str.split("\n")
    graph_str = ""
    for line in code_lines:
        if "add_edge" in line:
            line = line.split("(")[1][:-1]
            fields = line.split(", \"")
            source = fields[0].strip()
            edge_type = fields[1].strip()
            target = fields[2].strip()
            graph_str += f"({source}; {edge_type}; {target})"
    graph_str = graph_str.replace("\"", "")
    graph_str = graph_str.replace(";", "; ")
    graph_str = graph_str.replace("  ", " ")
    return {
        "belief": belief,
        "argument": argument,
        "stance": stance,
        "graph": graph_str,
    }


def _to_camel_case(title):
    """官方 to_camel_case：尾句号剥除 + 逐 token capitalize 连接。"""
    if "." == title[-1]:
        title = title[:-1]
    title_tokens = title.split(" ")
    title_camel_case = ""
    for token in title_tokens:
        title_camel_case += token.capitalize()
    return title_camel_case


def _to_snake_case(name):
    """官方 to_snake_case：尾句号剥除 + [\\s\\W]->'_' + lower + strip。"""
    if name[-1] == ".":
        name = name[:-1]
    name = re.sub(r'[\s\W]', '_', name)
    return name.lower().strip()


def _from_snake_to_normal_str(snake_str):
    """官方 from_snake_to_normal_str：split('_') 后 ' '.join。"""
    return " ".join(snake_str.split("_"))


def solve(mode="parse", graph_str=None, belief=None, argument=None,
          stance=None, prompt_part_only=False, code_str=None,
          src_dest=None, text=None, case=None):
    """统一入口（判题协议）：按 mode 分发到 parse / to_python / to_graph /
    roots / case。"""
    if mode == "parse":
        _check_str(graph_str, "graph_str")
        nodes, from_to, from_to_typed = _get_nodes(graph_str)
        return {"nodes": nodes, "from_to": from_to,
                "from_to_typed": dict((k, [tuple(e) for e in v])
                                      for k, v in from_to_typed.items())}
    if mode == "to_python":
        _check_str(belief, "belief")
        _check_str(argument, "argument")
        _check_str(stance, "stance")
        _check_str(graph_str, "graph_str")
        _check_bool(prompt_part_only, "prompt_part_only")
        return _graph_to_python(belief, argument, stance, graph_str,
                                prompt_part_only)
    if mode == "to_graph":
        _check_str(code_str, "code_str")
        return _python_to_graph(code_str)
    if mode == "roots":
        if not isinstance(src_dest, dict):
            raise ValueError("src_dest must be a dict")
        for key, dests in src_dest.items():
            if not isinstance(key, str) or not isinstance(dests, list) \
                    or any(not isinstance(d, str) for d in dests):
                raise ValueError("src_dest must map str -> list[str]")
        return _root_nodes(src_dest)
    if mode == "case":
        _check_str(text, "text")
        if case not in CASES:
            raise ValueError(f"case must be one of {CASES}")
        if case == "camel":
            return _to_camel_case(text)
        if case == "snake":
            return _to_snake_case(text)
        return _from_snake_to_normal_str(text)
    raise ValueError(f"Unsupported mode {mode!r}")
