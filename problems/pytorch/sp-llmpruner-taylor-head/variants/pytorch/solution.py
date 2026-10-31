import torch


def solve(wq, wk, wv, wo, gwq, gwk, gwv, gwo, head_dim,
          mode="param_first", group_reduction="sum", num_prune=1):
    """LLM-Pruner：多头注意力的组粒度一阶泰勒头重要性。

    上游 horseee/LLM-Pruner 的 hf_llama_pruner.py::TaylorImportance（attention 分支）：
      salience = weight * grad（逐元素一阶泰勒）
      param_first : o_proj 按 out 方向 |salience|.sum(1)；q/k/v 按 in 方向 |salience|.sum(0)
      vectorize   : o_proj 先 sum(1) 再 abs；q/k/v 先 sum(0) 再 abs（符号抵消）
      o 的 d 长向量按头分块求和、q/k/v 的块内向量求和 → 每头标量；
      四个贡献按 group_reduction（sum/mean）归并后升序 stable 排序，
      取最小的 num_prune 个头移除（并列取小索引）。
    """
    mats = {"q": wq, "k": wk, "v": wv, "o": wo}
    grads = {"q": gwq, "k": gwk, "v": gwv, "o": gwo}
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim < 1:
        raise ValueError("head_dim 必须是正整数")
    if mode not in ("param_first", "vectorize"):
        raise ValueError("mode 必须是 'param_first' 或 'vectorize'")
    if group_reduction not in ("sum", "mean"):
        raise ValueError("group_reduction 必须是 'sum' 或 'mean'")
    if isinstance(num_prune, bool) or not isinstance(num_prune, int):
        raise ValueError("num_prune 必须是 int")
    if num_prune < 0:
        raise ValueError("num_prune 必须 >= 0")

    shapes = {n: tuple(m.shape) for n, m in mats.items()}
    dtypes = {n: m.dtype for n, m in mats.items()}
    gshapes = {n: tuple(m.shape) for n, m in grads.items()}
    gdtypes = {n: m.dtype for n, m in grads.items()}
    if len(set(shapes.values())) != 1 or len(set(gshapes.values())) != 1:
        raise ValueError("四个权重与四个梯度必须同形状")
    if shapes["q"] != gshapes["q"]:
        raise ValueError("权重与梯度形状必须一致")
    if len(set(dtypes.values())) != 1 or len(set(gdtypes.values())) != 1:
        raise ValueError("权重与梯度必须同 dtype")
    if dtypes["q"] != gdtypes["q"] or dtypes["q"] not in (torch.float32, torch.float64):
        raise ValueError("权重与梯度必须同 dtype 且为 float32/float64")
    if wq.dim() != 2:
        raise ValueError("权重必须是二维 (out, in)")

    d_out, d_in = wq.shape
    if d_in % head_dim != 0 or d_out % head_dim != 0:
        raise ValueError("in_features 与 out_features 都必须能被 head_dim 整除")
    n_heads = d_in // head_dim
    if num_prune > n_heads:
        raise ValueError("num_prune 不能超过头数")
    if d_out != d_in:
        raise ValueError("MHA 要求 out_features == in_features（o_proj 行块与 q/k/v 列块对齐）")

    wq64, wk64, wv64, wo64 = (m.to(torch.float64) for m in (wq, wk, wv, wo))
    gq64, gk64, gv64, go64 = (g.to(torch.float64) for g in (gwq, gwk, gwv, gwo))

    sal_q, sal_k, sal_v, sal_o = (w * g for w, g in
                                  ((wq64, gq64), (wk64, gk64), (wv64, gv64), (wo64, go64)))

    if mode == "param_first":
        o_rows = sal_o.abs().sum(dim=1)                    # (d,)
        q_cols = sal_q.abs().sum(dim=0)                    # (d,)
        k_cols = sal_k.abs().sum(dim=0)
        v_cols = sal_v.abs().sum(dim=0)
    else:  # vectorize：先求和再取绝对值
        o_rows = sal_o.sum(dim=1).abs()
        q_cols = sal_q.sum(dim=0).abs()
        k_cols = sal_k.sum(dim=0).abs()
        v_cols = sal_v.sum(dim=0).abs()

    contribs = []
    for scores in (o_rows, q_cols, k_cols, v_cols):
        per_head = scores.view(n_heads, head_dim).sum(dim=1)   # 头块内求和
        contribs.append(per_head)
    stacked = torch.stack(contribs, dim=0)                     # (4, n_heads)
    if group_reduction == "sum":
        head_importance = stacked.sum(dim=0)
    else:
        head_importance = stacked.mean(dim=0)

    order = torch.sort(head_importance, stable=True)[1]
    pruned_heads = sorted(int(i) for i in order[:num_prune].tolist())

    return {
        "head_importance": head_importance,
        "per_source": stacked,            # (4, n_heads) o/q/k/v 各自的头块贡献
        "pruned_heads": pruned_heads,
        "n_heads": n_heads,
    }
