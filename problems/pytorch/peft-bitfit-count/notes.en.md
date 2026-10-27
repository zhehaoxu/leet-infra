# PEFT: BitFit Bias-Only Tuning Parameter Account — Algorithm Notes

## 1. Provenance and core idea

Ben Zakee & Awadalla (ACL 2022, arXiv:2106.10199): fine-tune only the biases. Three intuitions: (1) bias parameters are naturally sparse (well under one in a thousand in BERT-scale models); (2) biases are the "translation degrees of freedom" of activations, sensitive to distribution shift — the domain adaptation literature had already shown that tuning only biases can substantially align distributions; (3) implementation is free. On BERT/GLUE it approaches full fine-tuning, and is even more stable in few-shot and medium-data regimes (full fine-tuning overfits). BitFit is orthogonal to the prompt-family methods and stackable.

## 2. Per-layer parameter account (judged points)

| Layer type | Total params | BitFit trainable | Note |
| --- | --- | --- | --- |
| Linear(\(a\to b\), bias=True) | \(ab+b\) | \(b\) | bias is the **output dim** |
| LayerNorm(\(a\)) | \(2a\) | \(a\) | only \(\beta\); \(\gamma\)(weight) frozen |
| Linear(\(a\to b\), bias=False) | \(ab\) | \(0\) | no bias, nothing trainable |

`check_example`'s table: (0,8,4)+(1,4,0)+(2,4,6)+(0,6,6) → total \(36+8+24+42=110\), trainable \(4+4+0+6=14\), ratio \(=14/110\approx 0.1273\).

Traps (all pinned by assertions in the tests):

- **LayerNorm's \(\gamma\) is not a bias**: counting all \(2a\) as trainable is the most common mistake — `check_ln_beta_only` pins trainable \(=\Sigma a\) and total \(=2\Sigma a\) with an all-LayerNorm table.
- **Bias is the output dim**: Linear(8→4) has 4 biases, not 8 (covered by example row 1).
- **bias=False Linear contributes 0**: `check_no_bias_linear` asserts ratio \(=0\), trainable \(=0\) with an all-kind-2 table.

## 3. Division of labor within the collection

- `peft-prompt-tuning`/`peft-prefix-tuning`: the account is over "newly added learnable tensors"; BitFit's account is over a "base-model subset" — both conventions must be mastered.
- The `peft-lora-forward` family: LoRA's account is \(r(d_\text{out}+d_\text{in})\).
- This problem: pure parameter audit, no forward — it tests the precision of your memory of Transformer layer-type parameter structures.

## 4. Verification approach

`check_example` hand-computed as above; `check_matches_oracle` compares a random 20-layer table against an independent dict-dispatch recomputation (exact integer comparison + ratio tolerance \(10^{-5}\)); the two specialized checks each cover a single layer type.
