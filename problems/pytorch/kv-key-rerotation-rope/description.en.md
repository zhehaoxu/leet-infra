# KeyRerotation: RoPE Re-rotation After Eviction

## Problem Description

RoPE encodes "absolute position" as a rotation angle applied to key vectors. After KV compression **drops intermediate tokens**, the remaining keys still carry the rotation angles of their **original positions** — yet in the new cache they are treated as consecutive positions $0,1,2,\dots$, so the relative relationships become wrong. **KeyRerotation** (see FINCH, [arXiv:2407.09112](https://arxiv.org/abs/2407.09112); also what HF SinkCache does) applies, after the gather, an additional rotation of $(q-p)$ to each kept key so it "appears to sit" at its new compressed position $q$.

## Rules

Given `keys` of shape $(H, L, D)$ (**already RoPE-rotated at their respective original positions**), `indices` of shape $(H, n\_kept)$ (which original positions each head keeps, long), and `inv_freq` of shape $(D/2,)$: return the re-rotated keys of shape $(H, n\_kept, D)$.

For head $h$, new position $i\in[0,n\_kept)$, and its original position $p=\text{indices}_{h,i}$:

1. $delta = i - p$
2. Frequency angles $freqs_i = delta \cdot \text{inv\_freq}\ \ (D/2\text{ dims}),\quad emb_i = [\,freqs_i,\ freqs_i\,]\ \ (\text{concatenated into } D)$
3. Take the gathered $k = \text{keys}_{h,p}$; define $\text{rotate\_half}(k)$ as: the first half takes $-k[t+D/2]$, the second half takes $k[t-D/2]$
4. $\text{out}_{h,i} = k \odot \cos(emb_i) + \text{rotate\_half}(k) \odot \sin(emb_i)$

## Example

```
H=1, L=5, D=4, inv_freq=[1.0, 0.1], indices=[[0, 2, 4]]
keys[0] = [[1.0,0.5,0.2,0.8], [0.3,0.7,0.1,0.4], [0.9,0.2,0.6,0.5],
           [0.4,0.4,0.4,0.4], [0.7,0.1,0.3,0.9]]

i=0, p=0 → delta=0 → cos=1, sin=0 → unchanged [1.0, 0.5, 0.2, 0.8]
i=1, p=2 → delta=-1, k=[0.9,0.2,0.6,0.5], freqs=[-1.0,-0.1]
          emb=[-1.0,-0.1,-1.0,-0.1]
          out=[0.9912, 0.2489, -0.4331, 0.4775]
i=2, p=4 → delta=-2, k=[0.7,0.1,0.3,0.9] → out=[-0.0185, 0.2768, -0.7614, 0.8622]

out[0] = [[1.0, 0.5, 0.2, 0.8],
          [0.9912, 0.2489, -0.4331, 0.4775],
          [-0.0185, 0.2768, -0.7614, 0.8622]]
```

## Constraints

- CPU float64 tolerance $10^{-5}$; $D$ is even and `inv_freq` has length exactly $D/2$.
- Each row of `indices` is **strictly ascending**, with values in $[0, L)$; different heads may have different `indices`.
- $delta$ may be negative; the evenness of $\cos$ and oddness of $\sin$ automatically handle reverse rotation.
