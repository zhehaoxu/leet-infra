# Notes: LoRA-SB's Sign-Estimated Gradient SVD Initialization

## Source

- Paper: **LoRA-SB: Initialize LoRA Using Sign-estimation for Efficient LLM Fine-tuning** (arXiv:2411.19557).
- Official implementation: `CERT-Lab/LoRA-SB` (utils/svd_utils.py, utils/initialization_utils.py,
  utils/gradient_utils.py, utils/latent_utils.py).
- Motivation: the PiSSA/LoRA-GA family takes principal components from the weight SVD but still
  needs the first training steps to "align" them; LoRA-SB builds \(B@A\approx\widehat{\Delta W}\)
  directly from a **single small-sample gradient pass**, and the r_squared mode squeezes all
  trainable parameters into \(r^2\).

## Line-by-line translation of the official implementation

**Sign update** (end of gradient_utils.py, verbatim):

```python
for name, grad in running_grads_sum.items():
    processed_grads[name] = (-1 * lr * torch.sign(grad))   # sign(0)=0, magnitudes all lr
```

**svd_grad** (svd_utils.py, verbatim):

```python
input_matrix_float = input_matrix.to(torch.float32)   # official caller passes weight.T -> (in,out)
torch.manual_seed(42)                                  # fixed seed (judging domain: seed argument)
R = torch.randn(size, rank, ...)                       # dummy! never used in computation
U, s, V = torch.svd_lowrank(input_matrix_float, q=rank, niter=n_iter)
return U, diag_s, V.T                                  # "Vrt" = V.T
```

```python
# Quirk: R consumes the RNG stream BEFORE svd_lowrank; the dummy draw advances the stream by
# size*rank normal samples -> a different internal projection -> a bit-different SVD result.
# Removing the R line (m-sv-dummy) and changing the seed (m-sv-seed) are caught by value anchors.
```

**Split and assembly** (initialization_utils.py, verbatim):

```python
final_enc = torch.tensor(Ur @ sr, ...)   # (in, r)   -> .T on replace -> A_w (r,in)
final_dec = torch.tensor(Vrt, ...)       # (r, out)  -> .T on replace -> B_w (out,r)
# half_init_dec: kaiming_uniform_(matrix[ceil(rows/2):, :], a=sqrt(5)) on dec's lower rows, pre-transpose
# random_init:  kaiming(enc) first, then kaiming(dec) — one global stream; the order is a contract
# B_w@A_w = V diag(s) U^T ≈ sign_update (out,in); flat-spectrum sign matrices make a coarse recon by design
```

**r_squared forward** (latent_utils.py, verbatim):

```python
result += (
    self.lora_B[self.active_adapter[0]](
        self.default_lora_latent_mapping(
            self.lora_A[self.active_adapter[0]](self.lora_dropout[...](x))
        )
    ) * self.scaling[self.active_adapter[0]]
)
# get_delta_weight: output_tensor = transpose(B @ M @ A, fan_in_fan_out) * scaling
# grad mode: target.default_lora_latent_mapping.weight.data = Sr (diag of singular values)
#            A/B requires_grad = False — frozen SVD factors, only the r*r mapping trains
# weight mode(find_and_initialize): M is initialized by init_module_weights(sigma=1e-5), i.e. normal_
# non-r_squared: init_module_weights(target.lora_A.default, sigma=1e-5) — near-zero A
```

## Exclusions and red lines in the judging domain

- **Excluded**: `Accelerator`/distributed sync, `OffloadContext`, bfloat16 device shuffling,
  tqdm — pure engineering.
- **Excluded**: the `get_record_gradient_hook` gradient-accumulation hook (no backward in the
  judging domain).
- **Kept**: the sign compression, the full svd_lowrank flow (including the dummy-RNG quirk),
  the kaiming a=√5 accounts, and the latent forward/merged equivalence.
- **Anchoring**: s/enc/dec fixtures are anchored bit-for-bit (float32 pipeline determinism
  verified); `torch.equal` compares across dtypes by value and the anchor itself IS the exact
  double image of a float32 value, so the dtype contract needs an explicit dtype assertion.

## Key quirks (verified line-by-line against the source)

- **The dummy R consumes RNG**: deleting that line changes the SVD result — the judging domain
  tests it directly as "not equal" between the `svd_grad` and `svd_grad_nodummy` paths.
- **fan_in looks at the pre-transpose matrix**: kaiming fan_in of enc (in,r) is r (not d_in);
  of dec (r,out) is d_out. bound \(=\sqrt{6/(6\cdot fan\_in)}=1/\sqrt{fan\_in}\).
- **The RNG footprint of half_init**: half_init+random_init differs from pure random_init
  (half init already advanced the stream), yet random_init's bound assertions still hold.
- **Memory-order uniform_ on non-contiguous inputs**: svd_grad returns dec=V.T, a non-contiguous
  view whose uniform_ fill follows memory order -> a different RNG order than a contiguous
  replay. The order-replay test therefore uses independent contiguous inputs.
- **torch.sgn == torch.sign on reals**: mutation m-su-zero removed as a true equivalent.
- **torch.equal compares by value across dtypes**: the dtype contract needs an explicit
  `.dtype == torch.float64` assertion.

## Mutation testing (26/26, 1 equivalent exclusion)

- m-su-zero (sign->sgn): a true equivalent on real tensors, removed.
- 26 mutations cover: sign-update scaling/linearization, seed/dummy/niter/q/V.T/enc/dtype/rank
  validation, half-row choice/ceil/order/transpose/fan_in/bound/clone, the latent three-stage
  chain/scaling/merge, diag/near-zero std/zero init.

## Boundary with existing problems

- **TinyLoRA (795)**: SVD from the weight W, sqrt(S) split evenly, R = a fixed random-projection
  combination (v trainable). LoRA-SB: SVD from a **sign gradient estimate**, the asymmetric
  U·s/Vᵀ split, M = diag(Sr) as the trainable init.
- **PVeRA (804)**: kaiming a=0 over the full matrix; here a=√5 on the lower rows plus a full
  re-init and an order contract.
- No existing problem uses svd_lowrank or a sign-update account (grep-verified).
