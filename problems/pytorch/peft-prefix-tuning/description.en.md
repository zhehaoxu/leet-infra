# PEFT: Prefix-Tuning Virtual KV Prefix Attention

## Problem

**Prefix-tuning** (Li & Liang, ACL 2021, arXiv:2103.10385) concatenates a **learnable virtual prefix** on the KV side of attention: without changing any base parameters, it trains only the \(K,V\) vectors of \(L\) virtual tokens, letting the real tokens' queries "look at" this prefix:

\[\text{attn}=\text{softmax}\!\Bigl(\frac{Q\,[P_k;K]^{\!\top}}{\sqrt d}\Bigr)[P_v;V]\]

where \([P_k;K]\), \([P_v;V]\) are concatenations with the **prefix in front**: \(P_k,P_v\in\mathbb R^{L\times d}\), \(K,V\in\mathbb R^{S\times d}\), \(Q\in\mathbb R^{N_q\times d}\).

The trainable parameters have two implementations (the two forms of PEFT's `PrefixEncoder`, hidden width \(h=d\)):

- **Direct form**: \(P_k,P_v\) merged into one embedding table — \(n_\text{direct}=2Ld\).
- **Reparameterization form** (recommended by the paper; the MLP stabilizes optimization): first train an \(L\times d\) embedding, then generate the KV through \(\text{Linear}(d\to h)\to\tanh\to\text{Linear}(h\to 2Ld)\):

\[n_\text{reparam}=\underbrace{Ld}_{\text{embedding}}+\underbrace{(dh+h)}_{\text{first Linear}}+\underbrace{(h\cdot 2Ld+2Ld)}_{\text{second Linear}}\]

## Task

Implement single-head attention forward with a virtual prefix, and report the trainable parameter counts of both forms.

- Inputs: \(P_k,P_v\in\mathbb R^{L\times d}\) (\(L\ge 0\)), \(K,V\in\mathbb R^{S\times d}\), \(Q\in\mathbb R^{N_q\times d}\).
- Output \((\texttt{attn},\ \texttt{n\_direct},\ \texttt{n\_reparam})\): \(\texttt{attn}\in\mathbb R^{N_q\times d}\) (float32), the two parameter counts are integers.

**Convention**: compute in float64 internally, output float32; softmax over the last dimension of the scores; \(h=d\); no ready-made PEFT/transformers library. Tolerance \(10^{-5}\).

## Hints

- Concatenation order: prefix in front, real KV behind — the wrong order keeps the parameter counts right but the numbers wrong.
- With \(L=0\) it must degenerate to standard attention \(\text{softmax}(QK^\top/\sqrt d)V\) (use it as a self-check).
- Count only weights + biases in the reparameterization MLP; Tanh has no parameters; it has **more** parameters than the direct form, and the benefit is optimization stability.
