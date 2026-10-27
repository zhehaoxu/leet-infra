# PEFT: Prefix-Tuning Virtual KV Prefix Attention — Algorithm Notes

## 1. Provenance and core idea

Li & Liang (ACL 2021, arXiv:2103.10385) faced two expensive adaptation routes: full fine-tuning keeps one weight copy per task, and adapters insert serial modules between layers at extra latency. Prefix-tuning's insight is that **attention itself is the conditioning injection point** — give every layer's attention the K,V of \(L\) extra virtual tokens, and the real tokens' attention automatically learns to "consult the prefix", soft-prompting the model at generation time. On GPT-2 table-to-text, BERT XSum and similar tasks it reaches full fine-tuning quality with about 0.1% of the parameters, and overtakes it in low-data regimes.

Difference from prompt-tuning (a common interview question): prompt-tuning concatenates the prefix only at the **input embedding layer** (it only affects layer 0's input), while prefix-tuning attaches an independent prefix to the attention of **every layer** — more expressive, heavier to implement.

## 2. The reparameterization MLP (source-level convention)

Training \(P_k,P_v\) directly optimizes poorly (paper §4.1: reparameterization is clearly more stable in low-data regimes), so one trains a small tensor first and writes it back through an MLP. PEFT's `PrefixEncoder` reparameterization form (`prefix_tuning/model.py`):

```python
self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
self.transform = nn.Sequential(
    nn.Linear(token_dim, encoder_hidden_size),
    nn.Tanh(),
    nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
)
```

Parameter account (PEFT default \(h=d\)):

- embedding: \(L\cdot d\)
- first Linear: \(d\cdot h+h\)
- second Linear: \(h\cdot(2Ld)+2Ld\)

Total \(n_\text{reparam}=Ld+dh+h+2Ldh+2Ld\) (here \(h=d\)); direct form \(n_\text{direct}=2Ld\). **Note the reparameterization form has more parameters** — the benefit is optimization stability, not parameter saving (a common trap). `check_param_counts` asserts the formula with two \((L,d)\) pairs to prevent hardcoding.

## 3. Forward semantics (judged points)

- Concatenation happens on **both K and V** simultaneously, with the **prefix in front**: \([P_k;K]\), \([P_v;V]\). In a causal language model the real tokens' positions shift by \(L\) and every position sees the full prefix; this problem's single-round attention has no causal mask, but the concatenation order determines the semantics of the softmax weights (prefix rows pair with \(P_v\) rows).
- `check_prefix_zero_degenerates`: with \(L=0\), \([P_k;K]=K\) and it degenerates to standard attention \(\text{softmax}(QK^\top/\sqrt d)V\), while \(n_\text{direct}=0\) and \(n_\text{reparam}=d^2+d\) (the embedding vanishes, the MLP remains).
- `check_prefix_dominates`: when \(Q\) aligns strongly with \(P_k\) and is orthogonal to \(K\), the output approaches \(P_v\) — the minimal judgable verification of the intuition "virtual KV are learnable soft-prompt KV".

## 4. Division of labor within the collection

- `peft-prompt-tuning`: embedding-layer concatenation + per-task \(L\cdot d\) parameter account (input side).
- `peft-multi-adapter-batch`: in-batch multi-adapter routing on the LoRA side.
- This problem: KV prefix inside attention + the two-form parameter account.

## 5. Verification approach

`check_example` (\(d=L=S=N_q=1\)): \(P_k=2,P_v=3,K=5,V=7,Q=1\), logits \([2,5]\), attn \(=(3+7e^{3})/(1+e^{3})\approx 6.8103\), \(n_\text{direct}=2\), \(n_\text{reparam}=7\); `check_matches_oracle` compares against a pure-python softmax (float64 internally, float32 out, tolerance \(10^{-5}\)).
