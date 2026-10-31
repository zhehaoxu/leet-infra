# Quantization: PolarQuant Polar Pair Quantization

## Problem description

After RoPE, the channel pair $(d,\; d + D/2)$ of a key vector is really the two components of one 2-D vector: the rotation only changes that pair's angle, never its magnitude. PolarQuant exploits exactly this structure and folds the key into polar coordinates $(\phi, r)$ — the angle gets one low-bit code, the radius another, and the two are packed into a single integer `indices = (rho << tbits) + theta`; dequantising looks the angle and radius back up from the codes and takes $\cos/\sin$ to rebuild the key. Both the angle and the radius are min-max quantised along the token dimension, so every token in a group shares one $(\text{min},\text{scale})$ pair.

Upstream decodes with a GPU kernel; this problem follows the "PyTorch reference that checks our kernel" block the authors ship in `__main__` of `kernel4group.py`, matching its rounding direction, its denominator and its bin centre rule one by one.

## Rules

Write `keys` as $(B,H,L,D)$ with $D$ even and $D_h = D/2$; cut the token axis into $L/G$ groups of $G=$ `group_size` tokens. All internal arithmetic is float64, returned tensors use the input dtype.

1. **Pairing**: pair $d$ is $\big(\text{keys}[\ldots, d],\ \text{keys}[\ldots, D_h + d]\big)$ — the **front half with the back half** (the `rotate_half` partner of RoPE), not an odd/even interleave. Call them $x_{lo}$ and $x_{hi}$.
2. **Polar coordinates**: $\phi=\operatorname{atan2}(x_{hi}, x_{lo})\in(-\pi,\pi]$, and negative values get $+2\pi$ folded into $[0,2\pi)$; $r=\sqrt{x_{lo}^2+x_{hi}^2}$.
3. **Per-group min-max**: for each $(B,H,\text{group},d)$ take min/max over the **$G$ tokens of that group** (the reduced axis is tokens, not channels): $\text{scale}=(\max-\min)/2^{\text{bits}}$ — the denominator is $2^{\text{bits}}$, not $2^{\text{bits}}-1$. Angles use `tbits`, radii use `rbits`.
4. **Codes**: $\text{code}=\operatorname{clamp}\big(\lfloor (x-\min)/\text{scale}\rfloor,\ 0,\ 2^{\text{bits}}-1\big)$ — **floor**, not round-to-nearest; when $\text{scale}=0$ (constant group) the code is $0$. The angle code is $\theta$, the radius code is $\rho$.
5. **Reconstruction takes the bin centre**: $\hat{x}=(\text{code}+0.5)\cdot\text{scale}+\min$, hence $\hat\phi=(\theta+0.5)t_{\text{scale}}+t_{\min}$ and $\hat r=(\rho+0.5)r_{\text{scale}}+r_{\min}$, then
   $\text{recon}[\ldots,d]=\hat r\cos\hat\phi$ and $\text{recon}[\ldots,D_h+d]=\hat r\sin\hat\phi$.
6. **Packing**: $\text{indices}=(\rho \ll \text{tbits}) + \theta$, which lies in $[0,\,2^{\text{rbits}+\text{tbits}})$.

Return a dict with 6 keys: `phi`, `radii`, `theta`, `rho`, `indices` all have shape $(B,H,L/G,G,D_h)$ — the first two follow the input dtype, the last three are int64; `recon` is the $(B,H,L,D)$ dequantised rebuild and follows the input dtype.

## Task

Implement `solve(keys, group_size=128, rbits=4, tbits=4)`:

- `keys`: a 4-D float32 or float64 tensor $(B,H,L,D)$, read-only; $D$ must be even and $L$ must be a positive multiple of `group_size`;
- `group_size`, `rbits`, `tbits`: positive integers (`bool` and `4.0` are both illegal); $r_{\text{bits}}$ sizes the radius, $t_{\text{bits}}$ sizes the angle.

Illegal input raises `ValueError`. No argument may be modified in place.

## Example

$B=H=1$, $L=4$, $D=4$, `group_size=4`, `rbits=tbits=2`. The four tokens lay pair $d=0$ out as angles $0.1/0.9/1.6/2.0$ with radii $1/2/3/4$, while pair $d=1$ is constant at $(1,1)$. So $t_{\text{scale}}=(2.0-0.1)/4=0.475$ and $r_{\text{scale}}=(4-1)/4=0.75$: both $1.6$ and $2.0$ land in the last bin, giving $\theta=[0,1,3,3]$, while the four radii sit exactly on the bin centres, giving $\rho=[0,1,2,3]$.

```
keys    = [[[0.9950041652780258, 1.0, 0.09983341664682815, 1.0],
            [1.2432199365413290, 1.0, 1.5666538192549670, 1.0],
            [-0.0875985669038664, 1.0, 2.9987208091245150, 1.0],
            [-1.6645873461885700, 1.0, 3.6371897073027270, 1.0]]]

phi     = [[[[[0.0999999999999999, 0.7853981633974483],
              [0.9000000000000001, 0.7853981633974483],
              [1.6000000000000000, 0.7853981633974483],
              [2.0000000000000000, 0.7853981633974483]]]]]
radii   = [[[[[1.0, 1.4142135623730951], [2.0, 1.4142135623730951],
              [3.0, 1.4142135623730951], [4.0, 1.4142135623730951]]]]]
theta   = [[[[[0, 0], [1, 0], [3, 0], [3, 0]]]]]          # int64
rho     = [[[[[0, 0], [1, 0], [2, 0], [3, 0]]]]]          # int64
indices = [[[[[0, 0], [5, 0], [11, 0], [15, 0]]]]]        # (rho << 2) + theta

recon   = [[[1.2974299748897400, 1.0, 0.4553026029550115, 1.0],
            [1.4613318197185730, 1.0, 1.5427683924290140, 1.0],
            [-0.5477784488661208, 1.0, 2.8223330368611400, 1.0],
            [-0.6906771746572827, 1.0, 3.5585938290857850, 1.0]]]
```

The third token reconstructs at angle $0.1+3.5\times0.475=1.7625$ and radius $1.0+2.5\times0.75=2.875$, both strictly inside the group's span; the constant pair $d=1$ codes all-zero and reconstructs exactly to $(1,1)$.

## Constraints

- **Half-with-half pairing**: pair $d$ is $(d,\ D_h+d)$. Reading it as an odd/even interleave splits $(2,5)$ into $(1,2)$ and $(0,5)$, turning $\operatorname{atan2}(5,2)$ into $\operatorname{atan2}(2,1)$, so $\phi$, $\theta$ and `recon` are all wrong at once.
- **The denominator is $2^{\text{bits}}$**: the maximum element floors to $2^{\text{bits}}$ and only lands on $2^{\text{bits}}-1$ because of the clamp; with a denominator of $2^{\text{bits}}-1$ the top code stops at $2^{\text{bits}}-2$ and every bin in the group shifts by one.
- **The negative wrap is not decoration**: once $-0.1$ becomes $2\pi-0.1$ the group minimum is a different angle, and every code in the group shifts with it; $\operatorname{atan2}(0,3)=0$ is already non-negative and must not be wrapped to $2\pi$.
- **Reconstruction uses the bin centre**: $(\text{code}+0.5)\cdot\text{scale}+\min$. Using $\text{code}\cdot\text{scale}+\min$ collapses every bin onto its lower edge and puts the maximum back at $\max$.
- **The reduced axis is tokens**: min/max come from the $G$ tokens inside the group; reducing over $D_h$ mixes the two halves of one token into the same group.
- **A constant group gets code $0$**: upstream this is a $0/0$ NaN, and this problem defines the $\text{scale}=0$ group to code $0$ — then $(0+0.5)\cdot 0+\min=\min$, so the reconstruction equals the input exactly. Do not replace it with `clamp_min(eps)`.
- **The packing width follows `tbits`**: `indices` shifts $\rho$ left by `tbits` and adds $\theta$, so changing `tbits` changes the packing while changing `rbits` only moves the ceiling on $\rho$.
