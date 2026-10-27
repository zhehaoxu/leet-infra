# Diffusion: SDEdit Strength and the Truncated Start Step

## Problem

SDEdit turns edit strength into a knob by "noise up to some intermediate time, then sample back": a larger strength \(s\) destroys more and lands further from the original image. Given a cumulative-product table sorted by noise level, \(\bar\alpha_0>\bar\alpha_1>\dots>\bar\alpha_{T-1}\), the strength maps to a start index

\[ i=\left\lfloor s\cdot(T-1)\right\rfloor,\qquad x_i=\sqrt{\bar\alpha_i}\,x_0+\sqrt{1-\bar\alpha_i}\,\epsilon \]

and the sampler then only runs \(i\to i-1\to\dots\to0\) (that is \(i\) steps) — exactly the truncated start-step accounting. Note that \(\epsilon\) is an explicit input: this problem covers only strength → index → noising, with no sampling of randomness.

## Rules

Given `x0`, a same-shaped `noise`, a 1-D `alpha_bars` (strictly decreasing, entries in \((0,1]\)) and `strength∈[0,1]`, return `(x_noisy, i)`: `x_noisy` is a float32 tensor and `i` the int start index (which is also the number of remaining sampling steps).

## Examples

\[ \bar\alpha=[1.0,0.81,0.64,0.49,0.36],\ x_0=[1.0,-2.0],\ \epsilon=[0.5,1.0] \]

```
s=0.5 -> i=2, x_noisy=[ 1.100000, -1.000000]
s=1.0 -> i=4, x_noisy=[ 1.000000, -0.400000]
```

## Constraints

- `alpha_bars` must be 1-D, hold at least two entries, decrease strictly and stay inside \((0,1]\); `x0` and `noise` must share their shape.
- `strength` must lie in \([0,1]\); the index uses `floor` rather than rounding so that \(s=1\) lands exactly on the noisiest entry.
- **Degeneracies**: at \(s=0\), \(i=0\) and (when \(\bar\alpha_0=1\)) the output equals \(x_0\) element by element; at \(s=1\), \(i=T-1\) and the result is the noisiest level of the whole forward chain.
- The index is non-decreasing in strength; given the noise inverted from some \(x_i\), plugging in the matching strength must recover \(x_i\) exactly.
