"""ShadowPEFT starter — implement solve(mode, **kw) to pass all checks.

Reference: ShadowLLM/shadow-peft ``src/shadow_peft/modules.py`` (arXiv:2604.19254).

Modes to implement:
  inject        -> hidden + alpha * ((hidden - shadow) @ W_down) @ W_up
                   (W_down (d, k), W_up (k, d), einsum btd,dk / btk,kd)
  inject_zero_init      -> bool: W_up all zeros => output identical to hidden
  inject_sequence       -> per-layer list of injected hiddens
  update        -> h_in = LayerNorm(hidden); g = sigmoid(Linear(SiLU(Linear(h_in))));
                   ht = Linear(SiLU(Linear(h_in))); shadow + g * (ht - shadow)
  gate          -> the gate tensor g only
  gated_identity        -> bool: shadow + g*(ht - shadow) == (1-g)*shadow + g*ht
  param_count   -> {"injection", "update", "total"} trainable params
  init_injection        -> {"downs": normal(std=0.02), "ups": zeros}
"""
