"""EViT token 重组织账（纯 Python）。

对应官方源码（youweiliang/evit, master 分支）：
  evit.py::Attention.forward          —— cls_attn 打分 + topk 选 token
  evit.py::Block.forward              —— token 重组（gather / fuse）
  helpers.py::complement_idx          —— 补集计算

四个 mode（全部显式接收长度，不依赖任何隐式状态）：
  cls_attn_score   —— cls_attn 的打分口径（跨 head 均值 + 只取 cls 行 + 去掉 cls 列）
  select_tokens    —— topk 选 token + keep_rate 的 ceil 与触发条件
  fuse_token       —— 非保留 token 的加权求和（extra_token）与重组顺序
  reorganize       —— 一次完整的 Block 级重组（可选 fuse）

数值口径：能走精确有理数（Fraction）的部分一律走 Fraction；
最终所有出参统一转成 float。
"""

import math
from fractions import Fraction


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

_EPS = 1e-12


def _to_frac(value):
    """把输入转成精确有理数 Fraction。"""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric value")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("unsupported numeric type: %r" % (type(value),))


def _flat(value):
    """把嵌套列表展平成一维列表。"""
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _norm_list(value, name):
    """把入参归一化成一维 Fraction 列表。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    return [_to_frac(v) for v in _flat(value)]


def _norm_rows(value, name):
    """把入参归一化成「每行一个 list」的二维 Fraction 列表。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    rows = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            raise ValueError("%s must be a list of lists" % name)
        rows.append([_to_frac(v) for v in item])
    return rows


def _norm_heads(value, name):
    """把入参归一化成「每个 head 一个 N×N 矩阵」的三维 Fraction 列表。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    heads = []
    for h in value:
        if not isinstance(h, (list, tuple)):
            raise ValueError("%s must be a list of lists" % name)
        rows = []
        for r in h:
            if not isinstance(r, (list, tuple)):
                raise ValueError("%s must be a list of lists" % name)
            rows.append([_to_frac(v) for v in r])
        heads.append(rows)
    return heads


def _f(value):
    """出参统一转 float。"""
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


def _frow(row):
    return [_f(v) for v in row]


# --------------------------------------------------------------------------
# mode 1: cls_attn_score
# --------------------------------------------------------------------------


def cls_attn_score(attn):
    """cls_attn 的打分口径。

    官方实现（evit.py::Attention.forward）：
        cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]

    ★ 要点：
      1. 只取 attention 的【第 0 行】（cls token 作为 query）；
      2. 列从 1 开始 —— 【去掉 cls 自身那一列】；
      3. 然后沿 head 维求【算术平均】（不是 sum，也不是 max）。

    本题把 attn 展平成 [H, N, N]（单 batch），返回长度 N-1 的一维列表。

    返回：
      {"cls_attn": [...], "n_heads": int, "n_tokens": int}
    """
    heads = _norm_heads(attn, "attn")
    if not heads:
        raise ValueError("attn must not be empty")
    h = len(heads)
    n = len(heads[0])
    if n == 0:
        raise ValueError("attn rows must not be empty")
    for hr in heads:
        if len(hr) != n:
            raise ValueError("attn heads must all have the same row count")
        for r in hr:
            if len(r) != n:
                raise ValueError("attn rows must all have the same length")
    if n < 2:
        raise ValueError("attn must have at least 2 tokens (cls + 1 patch)")

    # 每个 head：取第 0 行（cls query），列 1..N-1（去掉 cls 列），再跨 head 求均值
    per_head = []
    for hr in heads:
        per_head.append(hr[0][1:])  # hr[0] 是 cls query 行，[1:] 去掉 cls 列
    n_patch = n - 1
    cls_attn = []
    for j in range(n_patch):
        s = Fraction(0)
        for h_i in range(h):
            s += per_head[h_i][j]
        cls_attn.append(s / h)

    return {
        "cls_attn": cls_attn,
        "n_heads": h,
        "n_tokens": n,
    }


# --------------------------------------------------------------------------
# mode 2: select_tokens
# --------------------------------------------------------------------------


def select_tokens(cls_attn, keep_rate, self_keep_rate=1.0, tokens=None):
    """topk 选 token 与 keep_rate 的 ceil 口径。

    官方实现（evit.py::Attention.forward）：
        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)
            _, idx = torch.topk(cls_attn, left_tokens, dim=1,
                                 largest=True, sorted=True)
            index = idx.unsqueeze(-1).expand(-1, -1, C)
            return x, index, idx, cls_attn, left_tokens
        return x, None, None, None, left_tokens

    ★ 要点：
      1. ★ 触发条件是
         `(self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)`
         —— Python 的 `and` 优先级高于 `or`，等价于
         `(self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)`。
         `self_keep_rate` 是【构造时】的值，`keep_rate` 是【调用时】的值，
         两者是**不同的变量**，必须都 < 1 才走 keep_rate 分支；
         但 `tokens is not None` 可以单独触发。
      2. ★ `left_tokens = math.ceil(keep_rate * (N - 1))` —— 【向上取整】，
         且基数是 `N - 1`（N 含 cls token）。
      3. ★ `tokens` 不为 None 时，`left_tokens` 被**直接覆盖**为 tokens
         （keep_rate 的 ceil 结果被丢弃）。
      4. ★ `left_tokens == N - 1` 时【提前返回】—— 不选 token、
         `index` / `idx` 均为 None（即使条件触发了）。
      5. `assert left_tokens >= 1` —— 不足 1 个是非法的。
      6. `torch.topk(..., largest=True, sorted=True)` —— 取分数最大的
         `left_tokens` 个，**返回值按分数降序排列**（不是下标升序）。

    返回：
      {"left_tokens", "triggered", "idx", "scores", "kept_scores",
       "complement", "n_tokens"}
    """
    scores = _norm_list(cls_attn, "cls_attn")
    if not scores:
        raise ValueError("cls_attn must not be empty")
    n_patch = len(scores)
    n_tokens = n_patch + 1  # N 含 cls token

    kr = _to_frac(keep_rate)
    self_kr = _to_frac(self_keep_rate)

    if tokens is not None:
        tok = int(tokens)
    else:
        tok = None

    # 触发条件：(self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)
    triggered = bool((self_kr < 1 and kr < 1) or (tok is not None))

    left_tokens = n_patch
    if triggered:
        left_tokens = int(math.ceil(float(kr * n_patch)))
        if tok is not None:
            left_tokens = tok

    if left_tokens == n_patch:
        # 提前返回：不选 token
        return {
            "left_tokens": left_tokens,
            "triggered": triggered,
            "idx": None,
            "scores": None,
            "kept_scores": None,
            "complement": None,
            "n_tokens": n_tokens,
        }

    if left_tokens < 1:
        raise ValueError("left_tokens must be >= 1, got %d" % left_tokens)

    # topk(largest=True, sorted=True)：按分数降序，并列按下标升序
    order = sorted(range(n_patch), key=lambda i: (-scores[i], i))
    idx = order[:left_tokens]
    kept_scores = [scores[i] for i in idx]

    # complement_idx(idx, N-1)：补集，按下标升序
    keep_set = set(idx)
    complement = [i for i in range(n_patch) if i not in keep_set]

    return {
        "left_tokens": left_tokens,
        "triggered": triggered,
        "idx": idx,
        "scores": kept_scores,
        "kept_scores": kept_scores,
        "complement": complement,
        "n_tokens": n_tokens,
    }


# --------------------------------------------------------------------------
# mode 3: fuse_token
# --------------------------------------------------------------------------


def fuse_token(non_cls, cls_attn, idx, complement):
    """非保留 token 的加权求和（extra_token）。

    官方实现（evit.py::Block.forward）：
        compl = complement_idx(idx, N - 1)
        non_topk = torch.gather(non_cls, dim=1,
                                 index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
        extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1),
                                dim=1, keepdim=True)

    ★ 要点：
      1. `non_cls` 是**去掉 cls 之后**的 patch 序列（长度 N-1），
         下标 i 对应原序列的第 i+1 个 token。
      2. `extra_token` 是【加权求和 sum】，**不是加权平均**：
         `sum_i(w_i * x_i)`，权重直接用 `cls_attn` 的值（softmax 后的小数），
         不做归一化。
      3. `compl` 是补集，**按下标升序**（官方 `complement_idx` 内部
         `torch.sort(..., descending=False)` 的结果）。
      4. 权重顺序必须与 `non_topk` 的顺序一致 —— 两者都按 `compl` 的顺序。

    返回：
      {"extra_token": [...], "non_topk": [...], "non_topk_attn": [...],
       "weight_sum": float}
    """
    rows = _norm_rows(non_cls, "non_cls")
    if not rows:
        raise ValueError("non_cls must not be empty")
    c = len(rows[0])
    for r in rows:
        if len(r) != c:
            raise ValueError("non_cls rows must all have the same length")
    if c == 0:
        raise ValueError("non_cls rows must not be empty")

    w = _norm_list(cls_attn, "cls_attn")
    # ★ cls_attn 的长度必须等于【patch 数】（= non_cls 的行数），
    #   不是通道数 —— 官方 cls_attn 是 [B, N-1]，每个 patch 一个权重。
    if len(w) != len(rows):
        raise ValueError(
            "cls_attn length %d does not match non_cls row count %d"
            % (len(w), len(rows))
        )

    idx_list = [int(i) for i in _flat(idx)]
    compl_list = [int(i) for i in _flat(complement)]

    # ★ gather：non_topk[j] 是 compl_list[j] 号 patch 的特征向量
    non_topk = [rows[i] for i in compl_list]
    non_topk_attn = [w[i] for i in compl_list]

    extra = [Fraction(0)] * c
    weight_sum = Fraction(0)
    for j, i in enumerate(compl_list):
        wi = non_topk_attn[j]
        weight_sum += wi
        vec = rows[i]
        for d in range(c):
            extra[d] += wi * vec[d]

    return {
        "extra_token": extra,
        "non_topk": non_topk,
        "non_topk_attn": non_topk_attn,
        "weight_sum": weight_sum,
    }


# --------------------------------------------------------------------------
# mode 4: reorganize
# --------------------------------------------------------------------------


def reorganize(hidden, idx, complement, fuse=True):
    """一次完整的 Block 级 token 重组。

    官方实现（evit.py::Block.forward）：
        non_cls = x[:, 1:]
        x_others = torch.gather(non_cls, dim=1, index=index)

        if self.fuse_token:
            compl = complement_idx(idx, N - 1)
            non_topk = torch.gather(non_cls, dim=1, index=compl...)
            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
            extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1),
                                    dim=1, keepdim=True)
            x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
        else:
            x = torch.cat([x[:, 0:1], x_others], dim=1)

    ★ 要点：
      1. 输出序列是 `[cls] + [按 idx 顺序 gather 的 patch] + ([extra] 若 fuse)`。
      2. ★ `extra_token` 放在【末尾】，不是最前 —— 常见的错误是把它插到 cls 后面。
      3. ★ gather 出来的 patch 保持 `idx` 的顺序（分数降序），
         **不是**原下标升序。
      4. `hidden` 是**已经加过注意力残差**的序列（官方在 `x = x + tmp`
         之后才做重组）。
      5. fuse 时输出长度 = 1 + len(idx) + 1；不 fuse 时 = 1 + len(idx)。
      6. `n_tokens` = 输出长度 - 1（去掉 cls）。

    返回：
      {"out": [...], "n_tokens", "kept_positions", "extra_token"}
    """
    rows = _norm_rows(hidden, "hidden")
    if not rows:
        raise ValueError("hidden must not be empty")
    c = len(rows[0])
    for r in rows:
        if len(r) != c:
            raise ValueError("hidden rows must all have the same length")
    if c == 0:
        raise ValueError("hidden rows must not be empty")
    # ★ hidden 是 [N][C]（N 个 token，C 个通道），第 0 行是 cls。
    n = len(rows)
    if n < 2:
        raise ValueError("hidden must have at least 2 tokens (cls + 1 patch)")

    idx_list = [int(i) for i in _flat(idx)]
    compl_list = [int(i) for i in _flat(complement)] if complement is not None else []

    cls_row = rows[0]
    patch_rows = rows[1:]  # ★ 去掉 cls 之后的 patch 序列（长度 N-1）

    # gather：按 idx 顺序（分数降序）取 patch
    kept = [patch_rows[i] for i in idx_list]

    out = [cls_row] + kept

    extra_token = None
    if fuse:
        # 加权求和（权重 = cls_attn，需外部传入时用 fuse_token）
        raise ValueError(
            "reorganize(fuse=True) requires cls_attn; use fuse_token first "
            "and pass its extra_token via the returned dict, or call "
            "reorganize with fuse=False"
        )

    return {
        "out": out,
        "n_tokens": len(out) - 1,
        "kept_positions": [i + 1 for i in idx_list],
        "extra_token": extra_token,
    }


def reorganize_with_extra(hidden, idx, extra_token, fuse=True):
    """带已算好的 extra_token 的重组（与 reorganize 配套）。

    返回：
      {"out": [...], "n_tokens", "kept_positions", "extra_token"}
    """
    rows = _norm_rows(hidden, "hidden")
    if not rows:
        raise ValueError("hidden must not be empty")
    c = len(rows[0])
    for r in rows:
        if len(r) != c:
            raise ValueError("hidden rows must all have the same length")
    if c == 0:
        raise ValueError("hidden rows must not be empty")
    n = len(rows)
    if n < 2:
        raise ValueError("hidden must have at least 2 tokens (cls + 1 patch)")

    idx_list = [int(i) for i in _flat(idx)]
    cls_row = rows[0]
    patch_rows = rows[1:]   # ★ 去掉 cls 之后的 patch 序列

    kept = [patch_rows[i] for i in idx_list]
    out = [cls_row] + kept

    extra = None
    if fuse:
        if extra_token is None:
            raise ValueError("extra_token is required when fuse=True")
        extra = _norm_list(extra_token, "extra_token")
        if len(extra) != len(cls_row):
            raise ValueError(
                "extra_token width %d does not match hidden width %d"
                % (len(extra), len(cls_row))
            )
        out = out + [extra]

    return {
        "out": out,
        "n_tokens": len(out) - 1,
        "kept_positions": [i + 1 for i in idx_list],
        "extra_token": extra,
    }


# --------------------------------------------------------------------------
# 单入口
# --------------------------------------------------------------------------


def solve(mode, **kw):
    if mode == "cls_attn_score":
        return cls_attn_score(kw["attn"])
    if mode == "select_tokens":
        return select_tokens(
            kw["cls_attn"], kw["keep_rate"],
            kw.get("self_keep_rate", 1.0), kw.get("tokens"),
        )
    if mode == "fuse_token":
        return fuse_token(
            kw["non_cls"], kw["cls_attn"], kw["idx"], kw["complement"],
        )
    if mode == "reorganize":
        if kw.get("fuse", False):
            return reorganize_with_extra(
                kw["hidden"], kw["idx"], kw.get("extra_token"), True)
        return reorganize(kw["hidden"], kw["idx"], kw.get("complement"), False)
    raise ValueError("unknown mode: %r" % (mode,))
