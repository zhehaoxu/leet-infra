# ShadowPEFT: coordinating adaptation across depth via shadow-state injection and gated updates

LoRA-style PEFT scatters adaptation into independent low-rank perturbations per Linear: no explicit task state is carried, updated, and reused across layers. ShadowPEFT (arXiv:2604.19254, official ShadowLLM/shadow-peft, merged into HF) instead performs **centralized layer-space refinement**: a depth-shared shadow state `s` couples with the frozen backbone layer by layer — filtering the difference signal at injection, then gating in fresh information after encoding.

## The two modules (official `modules.py`)

**ShadowInjectionModel** (applied before each base decoder layer except layer 0):

```python
delta = hidden - shadow            # (b, t, d) the difference carries the deviation direction
delta_t = delta @ W_down           # W_down: (d, k), N(0, 0.02²)
delta_t = delta_t @ W_up           # W_up: (k, d), zeros at init
hidden = hidden + alpha * delta_t
```

Difference-first is the key inductive bias: the bottleneck does not invent perturbations from scratch — it filters useful components out of the "gap between base and the shadow reference". The all-zero `W_up` init guarantees a zero-delta training start (same convention as LoRA).

**ShadowUpdateModel** (updates the shadow state after each base encoding):

```python
h_in = LayerNorm(hidden)                     # shared LN
g  = sigmoid(Linear(SiLU(Linear(h_in))))     # gate, bias-free throughout
ht = Linear(SiLU(Linear(h_in)))              # transform, no sigmoid, dropout mid-stack
shadow = shadow + g * (ht - shadow)          # equals (1-g)*shadow + g*ht
```

The GRU-style gate `g` decides how much of the shadow state is flushed by new information vs kept as accumulated task context; the official motivation is preventing shadow collapse and stabilizing optimization. **One deviation from the paper's formula, resolved by the code**: `T` has no sigmoid (the paper writes t=T(h_out), but the implementation's T is just a two-layer MLP); sigmoid lives only in the gate G.

## Parameter count

```
injection = L·(d·k + k·d)
update    = L·2·(d·gh + gh·d) + 2d    # 2d is the shared LayerNorm weight+bias
```

## What fundamentally differs from LoRA

The shadow module is **architecturally decoupled** from the backbone: detachable (shadow-only inference for edge deployment), pretrainable (a 0.5B model serving as the adapter for an 8B backbone), and parameter-shared across depth. The judging domain takes the injection/update pure-function cores: the difference-einsum chain, the gate algebra identity, the `W_up` zero-init invariant, parameter counts, and RNG replay.
