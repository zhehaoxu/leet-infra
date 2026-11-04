"""PEFT: Astra activation-space tail-eigenvector init (official PEFT PR #3725,
commit 73f9a1a, tuners/lora/astra.py + layer.py astra_init; arXiv:2602.19111).

Official account:
  1. Covariance accumulation (forward hook): flatten activations to (num_tokens, dim),
     cast float, normalize by the CALL's global max abs value, accumulate
     cov += Y^T Y with a per-call sample_count, then divide by the count.
  2. Eigendecomposition: S, V = torch.linalg.eigh(cov.float()); eigh is ASCENDING,
     so V[:, -r:] spans the top-r eigenspace. S is kept full length; only V is cropped.
  3. astra_init: lora_A = V^T @ W (r, in), lora_B = V (out, r);
     delta = scaling * lora_B @ lora_A; W -= delta (zero-delta residualization:
     W_new + scaling*B@A == W, and the adapter output equals x @ W^T at init).
  4. Validations: dtype in {float32, float16, bfloat16} (else TypeError);
     r <= out_features and V.shape == (out_dim, r) (else ValueError); NaN/Inf checks.
"""
import torch


def cov_accumulate(batches):
    """Official hook account: per-call normalization + accumulation + sample mean."""
    cov = None
    count = 0
    for b in batches:
        out = b.float()
        out = out.reshape(-1, out.size(-1))
        out = out / torch.max(out).abs()
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("Invalid value found in output, please check your output data.")
        c = out.t().matmul(out)
        if torch.isnan(c).any() or torch.isinf(c).any():
            raise ValueError("Invalid value found in covariance.")
        cov = c if cov is None else cov + c
        count += 1
    return cov / count, count


def astra_eigens(cov, r, out_dim=None):
    """Official collect_eigens_for_layer + crop: eigh (ascending), S kept full,
    V cropped to the last r columns (top-r eigenspace)."""
    if r <= 0:
        raise ValueError("r must be positive")
    covariance = cov.float()
    dim = covariance.size(0)
    S, V = torch.linalg.eigh(covariance)
    if S.size(0) != dim:
        raise ValueError(f"Matrix S size mismatch: {tuple(S.size())} vs. ({dim},)")
    if V.size(0) != dim or V.size(1) != dim:
        raise ValueError(f"Matrix V size mismatch: {tuple(V.size())} vs. ({dim}, {dim})")
    check = out_dim if out_dim is not None else dim
    if r > check:
        raise ValueError(
            f"Astra requires `r` <= out_features but got r={r} (max usable r is {check})."
        )
    S = S.clone()
    V = V[:, -r:].clone()
    if V.size(0) != dim:
        raise ValueError(f"Matrix V size mismatch: {tuple(V.size())} vs. ({dim},)")
    return S, V


def astra_init(weight: torch.Tensor, v_r: torch.Tensor, scaling: float = 1.0):
    """Official astra_init: A = V^T W, B = V, W -= scaling * B @ A."""
    if weight.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        raise TypeError(
            "Please initialize Astra under float32, float16, or bfloat16."
        )
    w32 = weight.to(torch.float32)
    out_dim, in_dim = w32.shape
    if torch.isnan(v_r).any() or torch.isinf(v_r).any():
        raise ValueError("Invalid value found in matrix V.")
    if v_r.size(0) != out_dim:
        raise ValueError(
            f"Matrix V size mismatch: {tuple(v_r.size())} vs. ({out_dim}, {v_r.size(1) if v_r.dim() > 1 else '?'})"
        )
    r = v_r.size(1)
    if r > out_dim:
        raise ValueError(
            f"Astra requires `r` <= out_features but got r={r} (max usable r is {out_dim})."
        )
    lora_a = (v_r.t() @ w32).contiguous()
    lora_b = v_r.contiguous()
    delta = scaling * lora_b @ lora_a
    w_new = w32 - delta
    return lora_a.to(weight.dtype), lora_b.to(weight.dtype), w_new.to(weight.dtype)


def astra_forward(x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor,
                  w_new: torch.Tensor, scaling: float = 1.0) -> torch.Tensor:
    """LoRA forward on the residualized base: scaling * (x A^T B^T) + x W_new^T."""
    return scaling * (x @ lora_a.t() @ lora_b.t()) + x @ w_new.t()


def solve(mode=None, **kw):
    if mode == "cov_accumulate":
        return cov_accumulate(kw["batches"])
    if mode == "eigens":
        return astra_eigens(kw["cov"], kw["r"], kw.get("out_dim"))
    if mode == "init":
        return astra_init(kw["weight"], kw["v_r"], kw.get("scaling", 1.0))
    if mode == "forward":
        return astra_forward(kw["x"], kw["lora_a"], kw["lora_b"],
                             kw["w_new"], kw.get("scaling", 1.0))
    raise ValueError(f"unknown mode {mode!r}")
