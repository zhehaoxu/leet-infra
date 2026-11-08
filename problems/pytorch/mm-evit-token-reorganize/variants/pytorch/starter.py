"""EViT token 重组织账 —— 骨架。

你需要实现四个 mode，由 solve(mode, **kw) 分派：

  cls_attn_score(attn)
      → {"cls_attn": [...], "n_heads": int, "n_tokens": int}

  select_tokens(cls_attn, keep_rate, self_keep_rate=1.0, tokens=None)
      → {"left_tokens": int, "triggered": bool, "idx": [...] | None,
         "scores": [...] | None, "kept_scores": [...] | None,
         "complement": [...] | None, "n_tokens": int}

  fuse_token(non_cls, cls_attn, idx, complement)
      → {"extra_token": [...], "non_topk": [[...]], "non_topk_attn": [...],
         "weight_sum": float}

  reorganize(hidden, idx, extra_token=None, fuse=True)
      → {"out": [[...]], "n_tokens": int, "kept_positions": [...],
         "extra_token": [...] | None}

关键口径：

- attn 按 [H][N][N] 给出：H 个 head，每个 head 是 N×N 的注意力矩阵
  （N 含 cls token，即 N = patch 数 + 1）。
- hidden 按 [N][C] 给出：N 个 token，C 个通道，第 0 行是 cls token。
- non_cls 按 [N-1][C] 给出：已经去掉 cls 之后的 patch 序列，
  下标 i 对应原序列的第 i+1 个 token。
- cls_attn 是长度 N-1 的打分行，下标 i 对应第 i 个 patch。
- 所有非法输入一律抛 ValueError。
- 能走精确有理数的部分走 fractions.Fraction；最终所有出参统一转 float。

四个容易写错的点：

1. cls_attn 只取 attention 的【第 0 行】（cls 作为 query），
   且列从 1 开始（【去掉 cls 自身那一列】），再跨 head 求【算术平均】。
2. 触发条件是 (self_keep_rate < 1 and keep_rate < 1) or (tokens is not None)
   —— self_keep_rate 是【构造时】的值，keep_rate 是【调用时】的值，
   是两个不同的变量，必须都 < 1 才走 keep_rate 分支。
3. left_tokens = math.ceil(keep_rate * (N - 1)) —— 【向上取整】，
   基数为 N-1；tokens 不为 None 时被【直接覆盖】；
   left_tokens == N - 1 时【提前返回】（idx 为 None）。
4. topk 的输出按【分数降序】排列（不是下标升序）；
   extra_token 是【加权求和】（不是加权平均），放在重组序列的【末尾】。

请在 solve 里完成实现，不要修改 test_cases.py。
"""

import math
from fractions import Fraction

_EPS = 1e-12


def _to_frac(value):
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
    if not isinstance(value, (list, tuple)):
        return [value]
    out = []
    for item in value:
        out.extend(_flat(item))
    return out


def _norm_list(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    return [_to_frac(v) for v in _flat(value)]


def _norm_rows(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError("%s must be a list" % name)
    rows = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            raise ValueError("%s must be a list of lists" % name)
        rows.append([_to_frac(v) for v in item])
    return rows


def _norm_heads(value, name):
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
    if isinstance(value, Fraction):
        return float(value)
    return float(value)


# --------------------------------------------------------------------------
# TODO: 实现下面四个函数
# --------------------------------------------------------------------------


def cls_attn_score(attn):
    """cls_attn 的打分口径（跨 head 均值 + 只取 cls 行 + 去掉 cls 列）。"""
    raise NotImplementedError("TODO: cls_attn_score")


def select_tokens(cls_attn, keep_rate, self_keep_rate=1.0, tokens=None):
    """topk 选 token 与 keep_rate 的 ceil 口径。"""
    raise NotImplementedError("TODO: select_tokens")


def fuse_token(non_cls, cls_attn, idx, complement):
    """非保留 token 的加权求和（extra_token）。"""
    raise NotImplementedError("TODO: fuse_token")


def reorganize(hidden, idx, extra_token=None, fuse=True):
    """一次完整的 Block 级 token 重组（可选 fuse）。"""
    raise NotImplementedError("TODO: reorganize")


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
            return reorganize(
                kw["hidden"], kw["idx"], kw.get("extra_token"), True)
        return reorganize(kw["hidden"], kw["idx"], kw.get("extra_token"), False)
    raise ValueError("unknown mode: %r" % (mode,))


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------


if __name__ == "__main__":
    print("skeleton loaded — implement the four modes above.")
