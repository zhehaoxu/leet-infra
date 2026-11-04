# Diffusion: FasterCache CFG-Cache Frequency Approximation Step (Delta State Machine + Low-Frequency Mask Reconstruction)

## Problem

The denoiser-side CFG-Cache of FasterCache: CFG sampling runs two forwards (uncond + cond) per timestep, but the `uncond − cond` output difference of adjacent steps varies slowly **in the frequency domain** — approximate this step's uncond branch with the previous step's frequency-domain delta and save half the batch forward. Implement `solve(model_output, state, current_timestep, unconditional_batch_skip_range=5, unconditional_batch_timestep_skip_range=(-1, 641), low_frequency_weight_update_timestep_range=(99, 901), high_frequency_weight_update_timestep_range=(-1, 301), alpha_low_frequency=1.1, alpha_high_frequency=1.1, is_guidance_distilled=False)`:

```text
# 1. Gate (open interval + modulo)
within = lo < current_timestep < hi                  # open interval (strict)
skip_uncond = within and iteration % N != 0 and not distilled

# 2. Non-skip step: full-batch forward (output 2B = uncond||cond)
uncond, cond = chunk(2, dim=0)
each BCFHW -> permute(0,2,1,3,4) -> flatten(0,1) gives (B*F, C, H, W)
_split_low_high_freq(x):
    fft2 -> fftshift (all dims) -> circular low-frequency mask
    radius = min(H, W) // 5, center (W//2, H//2)
    low = shifted * mask, high = shifted * ~mask
delta_low  = low(uncond)  - low(cond)                # complex tensors, stored in state
delta_high = high(uncond) - high(cond)

# 3. Skip step: run cond only (output B)
weight_low  = alpha_low  if lo_l < t < hi_l else 1.0 # each update open interval
weight_high = alpha_high if lo_h < t < hi_h else 1.0
delta_low, delta_high <- each * weight               # applied to the **stored** delta
uncond_freq = (delta_low + low(cond)) + (delta_high + high(cond))
uncond = real(ifft2(ifftshift(uncond_freq)))         # restore BCFHW layout
output = cat([uncond, cond], dim=0)                  # 2B
the skip step stores the **scaled** delta; non-skip steps recompute and overwrite
```

Return `dict(output, uncond_skipped, state)` with a four-key state `{iteration, batch_size, low_frequency_delta, high_frequency_delta}` (new dict, functional; batch_size recorded at the first non-skip step and kept, positive even).

## Key Semantics

- **All three gate conditions are required**: the timestep open interval, `iteration % N != 0` (iterations 0, N, 2N… run the full batch), and non-distilled. A distilled model always runs the full batch and the delta is never established.
- **The skip-step input is the conditional half (B)**: `batch * 2 == state.batch_size` must hold and the delta must be established (updated by a preceding full-batch step) — skipping directly on the first step raises an explicit error.
- **The weight acts on the stored delta**: `delta <- delta * weight` is a cumulative scaling of the state (official Equation 11); consecutive skips compound exponentially (×alpha^k), while a full-batch step recomputes and overwrites rather than continuing to scale.
- **The low-frequency mask is a circle with inclusive boundary**: `(x−cx)² + (y−cy)² <= r²`, r = min(H,W)//5, centered at (W//2, H//2) after fftshift; the low/high supports are complementary.
- **No pipeline step of the reconstruction can be dropped**: ifftshift restores the spectrum layout → ifft2 → take the real part → unflatten + permute restores BCFHW → uncond goes first in the cat.
- **The official `iteration > 0 and ...` is dead code** (0%N==0 always true) and is not included in the judgement domain.

## Judgement-domain Conventions

- The official hook wraps the denoiser forward with the weight from a callback; the judgement domain takes it explicitly, inlining the official default rule (alpha applies inside each update open interval, 1.0 outside).
- `model_output` is "the output actually executed this call": the conditional half (B) on skip steps, otherwise the full batch (2B).
- tensor_format is fixed "BCFHW" (the official BFCHW/BCHW branches and tuple-output branch are not included).
- The official code validates nothing; the judgement domain adds tensor/range/alpha/state four-key/skip preconditions (batch_size and delta established)/input-batch-vs-gate consistency checks.

## Example

The full non-distilled sequence (full batch 2B=4, conditional half B=2, N=5, t=400 inside the range): iter0 full-batch compute (delta = d0); iter1-4 conditional-half skip (uncond rebuilt in the frequency domain from `(d0·w + low_cond) + (d0·w + high_cond)`; at t=400, w_low=1.1, w_high=1.0; the state stores the scaled delta each time); iter5 full-batch compute (delta recomputed and overwritten). The `uncond_skipped` sequence is `[F,T,T,T,T,F]`.

Data scale: tensors `(2,1,2,8,8)`–`(4,1,2,8,8)` float32, N ≤ 8, alpha ∈ [0, 2].
