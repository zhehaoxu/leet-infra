# Deep Dive: PAG Dual-Scale Guidance and Its Two Windows

## Where it comes from

*Perturbed Attention Guidance* (arXiv:2403.17377). The implementation follows HuggingFace diffusers' `src/diffusers/guiders/perturbed_attention_guidance.py`: `forward` (L213–240) assembles the four branches, while `_is_cfg_enabled` (L258–274) and `_is_slg_enabled` (L277–289) define the two windows — the former uses `<= step <`, the latter keeps `step` strictly inside. The class reuses `SkipLayerGuidance`'s parameter names, which is why PAG's scale field is called `skip_layer_guidance_scale` and its window fields `skip_layer_guidance_start/stop` in the source.

## Key ideas and pitfalls

1. **The two windows do not share endpoint semantics**: CFG is \([\,lo, hi)\), PAG is \((lo, hi)\). Writing PAG as left-closed adds one extra perturbed-guidance step at `step = int(perturbed_start*N)` — with default parameters that is `step=0`, wrongly pulled into the PAG branch.
2. **Window arithmetic truncates**: `int(...)` on non-negative values equals `floor`, but `round(...)` or `ceil(...)` shifts every boundary. For very few steps (e.g. \(N=3\), `perturbed_stop=0.2`) the PAG window degenerates to empty and the branch never fires.
3. **"Off" has two sources**: outside the window, or a degenerate scale (\(s\approx 1\) disables CFG, \(s_{\text{skip}}\approx 0\) disables PAG). Both live in the same `and not isclose(...)` expression; splitting them into two independent flags misroutes the case that is simultaneously out-of-window and scale-degenerate.
4. **The PAG branch uses the perturbed prediction as its base**: PAG alone gives \(\epsilon_{\text{skip}}+s_{\text{skip}}(\epsilon_c-\epsilon_{\text{skip}})\), not a deviation from \(\epsilon_u\). Writing \(\epsilon_u + \dots\) produces a completely wrong answer inside the PAG-only window.
5. **With both branches on, the two deltas simply add over the same \(\epsilon_u\) base**: \(\epsilon_u+s\Delta_{\text{cfg}}+s_{\text{skip}}\Delta_{\text{pag}}\). Stacking the PAG term on top of the already-guided prediction (so its base becomes the guided one) is another common mistake and breaks the additivity assertions.
6. **A disabled branch may be `None`**: in diffusers the unused conditioning is never even inferred, so `pred_uncond` / `pred_cond_skip` are optional. The judge checks both directions: `None` is accepted when the branch is off, and raising is required when it is on.

## How the judge cross-checks

The reference solution computes windows and branches in float64, combines elementwise, then casts to float32. The `_oracle` in the tests reuses no tensor ops: windows are re-derived with python `int()` and `math.isclose`, and the combination is a per-element python scalar loop (reshape to `(batch, width)`, then flatten). Five structural assertion groups: four characteristic steps around the window edges (left endpoint, right endpoint, just inside, just outside), scale degeneracy (\(s=1\) / \(s_{\text{skip}}=0\) / both), the additivity identities `both - cfg == s_skip·Δ_pag` and `both - pag == s·Δ_cfg`, the `None` semantics of optional branches, and invalid inputs. Tolerance `allclose(atol=1e-5)`.

## Complexity

Elementwise \(O(N)\) with no reductions and no extra buffers; each of the three predictions is read once.
