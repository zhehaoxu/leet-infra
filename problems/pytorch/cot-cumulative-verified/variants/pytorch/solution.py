"""Cumulative Reasoning（arXiv:2308.04371, iiis-ai/cumulative-reasoning）累积态验证账。

Game of 24 官方实现（Game24/game24-openai.py solve() L537-590）把 proposer /
valid 判定 / verifier 三类 LLM 调用组织成一个**状态图生长循环**：每轮从已达
状态集合里按两阶段分布采样一个状态，proposer 给出一步算式与剩余数字串，
valid 判定合法后由 verifier 给出 sure / likely / impossible 三态结论——
likely 且新状态未见过才把状态连同父指针、步进与 forbidden 记录入图；sure
则沿父指针链回溯重构完整解题路径。本题把三类 LLM 输出全部显式入参，考纯账本。

solve(mode=...) 三个入口（官方逐条钉死）：

    mode="solve"（默认）：puzzle + rounds + trycnt -> (success, marker, steps)。
        rounds 每轮含 {chosen, next, remaining, judgement, verify, equation}；
        success 时 marker=轮下标、steps=回溯重构的中间步列表；失败时
        marker=trycnt、steps=[]。
    mode="state"：同参运行，返回中途账本快照
        {"records", "parent", "last_step", "forbidden"}。
    mode="dist"：index + n_states + trycnt -> 采样分布列表（前半程 p[0]=0.5
        的根偏置，其余均匀）。

单轮账本（官方顺序逐条钉死，任何异常静默跳过该轮——官方 try/except continue）：
    1. a = remaining.strip().split('\\n')[0].strip()（取首行）；
    2. 门 A：next.strip() 含 [^0-9+\\-*/.(),=\\s] 之外字符 -> 跳过；
    3. 门 B：next.strip() 或 remaining.strip() 无非空白（\\S）-> 跳过；
    4. judgement == 'Invalid' -> 跳过（只有 'Invalid' 拦，'Valid' 等放行）；
    5. verify == 'sure'：steps = [equation 首行] + [next.strip() + ' (left a)']
       + 沿父链回溯各步（每步拼 ' (left 状态)'），整体 reverse；
    6. verify == 'likely' 且 a 不在 records：forbidden[chosen] 追加
       next.strip()、forbidden[a]=[]、records 追加 a、parent[a]=chosen、
       last_step[a]=next.strip()；a 已在 records 则**整轮静默丢弃**；
    7. 其他 verify（impossible 等）无动作。
"""
import re

_BAD_CHAR = re.compile(r"[^0-9+\-*/.(),=\s]")
_HAS_NONSPACE = re.compile(r"\S")
ROUND_KEYS = ('chosen', 'next', 'remaining', 'judgement', 'verify', 'equation')


def _check_int(value, name, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _sample_dist(index, n_states, trycnt):
    """官方 p 向量构造：前半程根偏置（p[0]=0.5，其余 0.5/(n-1)），后半程均匀。"""
    p = [0.0] * n_states
    if index < 1 / 2 * trycnt:
        if n_states > 1:
            for j in range(n_states):
                p[j] = 0.5 / (n_states - 1)
            p[0] = 0.5
        else:
            p[0] = 1.
    else:
        for j in range(n_states):
            p[j] = 1. / n_states
    return p


def _run_rounds(puzzle, rounds, trycnt):
    """官方 solve() 主循环；返回 (records, last_step, parent, forbidden, result)。"""
    records = [puzzle]
    last_step = {}
    parent = {}
    forbidden = {puzzle: []}
    result = (False, trycnt, [])
    for i in range(trycnt):
        if i >= len(rounds):
            break
        rnd = rounds[i]
        try:
            tmp = rnd['chosen']
            next_step = rnd['next'].strip()
            a = rnd['remaining'].strip().split('\n')[0].strip()
            if _BAD_CHAR.search(next_step):
                continue
            if not _HAS_NONSPACE.search(next_step) or \
                    not _HAS_NONSPACE.search(rnd['remaining'].strip()):
                continue
            if rnd['judgement'] == 'Invalid':
                continue
            verify = rnd['verify']
            if verify == 'sure':
                tmp_steps = [rnd['equation'].strip().split('\n')[0].strip()]
                tmp_steps.append(next_step + f' (left {a})')
                t = tmp
                while t != puzzle:
                    tmp_steps.append(last_step[t] + f' (left {t})')
                    t = parent[t]
                tmp_steps.reverse()
                result = (True, i, tmp_steps)
                break
            elif verify == 'likely':
                if a not in records:
                    forbidden[tmp].append(next_step)
                    forbidden[a] = []
                    records.append(a)
                    parent[a] = tmp
                    last_step[a] = next_step
        except Exception:
            continue
    return records, last_step, parent, forbidden, result


def solve(mode="solve", puzzle=None, rounds=None, trycnt=None,
          index=None, n_states=None):
    """统一入口（判题协议）：按 mode 分发到 solve / state / dist。"""
    if mode == 'dist':
        _check_int(index, "index", 0)
        _check_int(n_states, "n_states", 1)
        _check_int(trycnt, "trycnt", 0)
        return _sample_dist(index, n_states, trycnt)
    if mode in ('solve', 'state'):
        if not isinstance(puzzle, str):
            raise ValueError("puzzle must be a str")
        _check_int(trycnt, "trycnt", 0)
        if not isinstance(rounds, (list, tuple)):
            raise ValueError("rounds must be a list")
        for rnd in rounds:
            if not isinstance(rnd, dict) or any(k not in rnd for k in ROUND_KEYS):
                raise ValueError(
                    "each round must be a dict with keys chosen/next/remaining/"
                    "judgement/verify/equation")
        records, last_step, parent, forbidden, result = \
            _run_rounds(puzzle, list(rounds), trycnt)
        if mode == 'state':
            return {'records': records, 'parent': parent,
                    'last_step': last_step, 'forbidden': forbidden}
        return result
    raise ValueError(f"Unsupported mode {mode!r}")
