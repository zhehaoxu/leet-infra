"""PEANuT starter — implement solve(mode, **kw) to pass all checks.

Reference: HF PEFT ``src/peft/tuners/peanut/layer.py`` (arXiv:2410.01870).

Modes to implement:
  delta_weight     -> (out, in) tensor: act(W^T @ A^T) through depth
                      encoder/decoder pairs with reverse-order residual adds,
                      then B, then * scaling, then transpose.
  forward          -> base_layer(x) + x @ delta.T
  param_count      -> {"peanut": out*r + r*out + 2*depth*r*r,
                       "lora": in*r + r*out, "saving": lora - peanut}
  init_factors     -> {"A": (r, out), "encoders": [depth x (r, r)],
                       "decoders": [depth x (r, r)], "B": (out, r)}
                      RNG order A -> encoders -> decoders -> B (zeros if
                      init_weights else kaiming a=sqrt(5)).
  zero_init_check  -> bool: with init_weights=True the merged delta is all zero
  residual_order   -> [(i, depth-1-i)] — the residual pairing the decoder loop uses
"""
