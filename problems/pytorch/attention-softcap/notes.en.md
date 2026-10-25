# Soft-Capping Explained

## What problem it solves

If the absolute magnitude of attention logits is uncontrolled, $\text{softmax}$ saturates into nearly one-hot: the largest entry monopolizes the probability,
the other gradients go to zero, and under bf16/fp16 it easily overflows, causing training instability or even loss spikes. Hard clipping with
`clamp(-cap, cap)` does bound the values, but **the gradient is 0 at the boundary**, cutting off information abruptly.

## Smooth capping

Gemma 2 uses a scaled `tanh` as a soft threshold:

$$ f(x) = cap \cdot \tanh(x / cap) $$

- For $|x| \ll cap$, $f(x)\approx x$ (nearly linear, no loss of information);
- As $x \to \pm\infty$, $f(x) \to \pm cap$ (bounded);
- It is smooth and differentiable everywhere, $f'(x)=1-\tanh^2(x/cap) \in (0,1]$, and the gradient at the boundary is non-zero.

Then apply the ordinary softmax. Note that **capping acts on the logits and happens before the softmax**.

## Common mistakes

- Order: cap with `tanh` first, then softmax; doing it the other way (softmax then cap) is completely wrong.
- The scaling is **divide by cap then multiply back by cap**: `cap*tanh(x/cap)`; omitting the inner `/cap` shifts the knee position.
- When `cap` is very large, $\tanh(x/cap)\approx x/cap$, so it degenerates to a plain softmax (regression case).
- It differs from temperature scaling: temperature does `logits/τ` before softmax (changing sharpness); soft-cap **bounds the upper end**, and the two can be combined.

## Applications

Gemma 2 uses `attn_logit_softcapping=50` on attention logits throughout and `final_logit_softcapping=30` on the output logits;
it is one of the key ingredients for its training robustness, especially under bf16 mixed precision where it prevents numerical blow-ups caused by extreme logits.
