# PEFT: Prompt-Tuning Soft-Prompt Embedding Concatenation — Algorithm Notes

## 1. Provenance and core idea

Lester et al. (EMNLP 2021, arXiv:2104.08691, *The Power of Scale for Parameter-Efficient Prompt Tuning*): prefix-tuning inserts a prefix at every layer and is heavy to implement; prompt-tuning cuts it down to just \(L\) learnable tokens at the input embedding layer. The key observation on the T5 family is the **scale effect**: the larger the model, the smaller the gap between prompt-tuning and full fine-tuning (nearly gone on T5-XXL), while each task needs only \(L\cdot d\) parameters (\(L=20,d=1024\) → about 20K parameters). In multi-task deployment the model exists once and each task holds one prompt table — the serving paradigm of "the prompt is the task identifier" starts here.

## 2. Forward and the pool form (source-level convention)

Prompt-tuning in HF transformers (`PromptEmbedding`): the model embedding is frozen, and the forward is

```python
inputs_embeds = torch.cat([prompt_embedding.expand(B, -1, -1),
                           embedding(input_ids)], dim=1)
```

prompt in front, original embeddings behind, sequence length becomes \(L+S_0\). This problem's multi-task pool form: all tasks' prompt rows live in one \(P\times d\) pool table, `task_ids[t]` is task \(t\)'s \(L\) row indices, and the forward is just a gather. Judged points:

- `check_gather_indices`: \(X[t,:L]\) equals `pool[task_ids[t]]` row by row — index semantics (not sorting, not a contiguous slice).
- `check_prepend_order`: \(X[t,L:]=E\) — prefix in front; the common implementation bug of "embeddings in front" is pinned.
- `check_param_account`: \(n_\text{params}=T\cdot L\cdot d\), unchanged when the pool size changes (\(P=6\to 9\)) — the "per-task exclusive view" of trainable parameters is decoupled from the storage form. Interview convention: prompt-tuning has \(L\cdot d\) trainable parameters per task; pool + indices is a serving storage optimization and does not change the account.

## 3. Comparison with prefix-tuning (easily confused)

| | prompt-tuning | prefix-tuning |
| --- | --- | --- |
| Concatenation site | input embedding layer only | KV of every layer's attention |
| Parameters per layer/task | \(L\cdot d\) (once) | direct \(2Ld\) (single layer), reparam more |
| Deep-layer effect | propagated only through layer 0 | injected directly at every layer |
| Multi-task pool | naturally supported | one set per layer per task, pooling complex |

## 4. Division of labor within the collection

- `peft-prefix-tuning`: KV-side prefix + the two-form parameter account.
- `peft-bitfit-count`: the parameter account of another minimal PEFT (bias subset).
- This problem: embedding-side prefix + gather routing + the per-task account.

## 5. Verification approach

`check_example` hand computation: a \(4\times2\) pool, the two tasks fetching rows 0/1 and 3/2, \(E=[[9,9]]\) → \(X_0\)'s first two rows are pool rows 0/1, \(X_1\)'s are rows 3/2, the trailing rows are both \([9,9]\), \(n=8\); `check_matches_oracle` compares against a pure-python double loop (float64 internally, float32 out, tolerance \(10^{-5}\)).
