import torch

_NEG_INF = -1e38  # 官方 kernel 的 -inf 代理常量，题面钉死


def solve(r, k, v, w, u, state=None):
    for name, t in (("r", r), ("k", k), ("v", v), ("w", w)):
        if not torch.is_tensor(t) or t.dim() != 2:
            raise ValueError(f"{name} must be a [T, C] tensor")
    if not torch.is_tensor(u) or u.dim() != 1:
        raise ValueError("u must be a [C] tensor")
    T, C = r.shape
    if k.shape != (T, C) or v.shape != (T, C) or w.shape != (T, C):
        raise ValueError("r/k/v/w shape mismatch")
    if u.shape[0] != C:
        raise ValueError("u channel mismatch")
    if T == 0 or C == 0:
        raise ValueError("empty input")

    rf = r.float()
    kf = k.float()
    vf = v.float()
    wf = w.float()
    uf = u.float()

    if state is None:
        a = torch.zeros(C)
        b = torch.zeros(C)
        p = torch.full((C,), _NEG_INF)
    else:
        if set(state.keys()) != {"a", "b", "p"}:
            raise ValueError("state must have keys a/b/p")
        a = state["a"].float().clone()
        b = state["b"].float().clone()
        p = state["p"].float().clone()
        if a.shape != (C,) or b.shape != (C,) or p.shape != (C,):
            raise ValueError("state channel mismatch")

    wkv_out = []
    p_trace = []
    for t in range(T):
        kt = kf[t]
        vt = vf[t]
        wt = wf[t]
        rt = rf[t]

        p_out = torch.maximum(p, uf)
        e1 = torch.exp(p - p_out)
        e2 = torch.exp(uf - p_out)
        wkv_t = (e1 * a + e2 * vt) / (e1 * b + e2)
        y_t = rt * wkv_t

        ww = p + wt
        p_new = torch.maximum(ww, kt)
        e1n = torch.exp(ww - p_new)
        e2n = torch.exp(kt - p_new)
        a = e1n * a + e2n * vt
        b = e1n * b + e2n
        p = p_new

        wkv_out.append(wkv_t)
        p_trace.append(p.tolist())

    wkv = torch.stack(wkv_out, dim=0)
    y = torch.stack([rf[t] * wkv[t] for t in range(T)], dim=0)
    return {
        "wkv": wkv,
        "y": y,
        "final_state": {"a": a, "b": b, "p": p},
        "p_trace": p_trace,
    }
