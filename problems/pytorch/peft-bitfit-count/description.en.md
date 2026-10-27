# PEFT: BitFit Bias-Only Tuning Parameter Account

## Problem

**BitFit** (Ben Zakee & Awadalla, ACL 2022, arXiv:2106.10199) is minimal PEFT: **fine-tune only the bias parameters** (or a subset of them), freeze everything else. On BERT-family models it matches full fine-tuning with a tiny trainable fraction (and overtakes it in few-shot regimes); the implementation is literally "turn off `requires_grad` on all weights".

The engineering question to answer is: given a list of layers, **how many parameters are trainable, how many in total, and what fraction**. "Bias" in a Transformer comes in three flavors by layer type:

- `nn.Linear(a→b)` (bias=True): the \(a\cdot b\) weights are frozen, **the \(b\) biases are trainable** — total \(ab+b\), trainable \(b\);
- `nn.LayerNorm(a)`: only \(\beta\) (the bias term, \(a\) of them) is trainable, \(\gamma\) (weight) is frozen — **total \(2a\), trainable \(a\)**;
- `nn.Linear(a→b)` (bias=False): no bias — **total \(ab\), trainable 0**.

## Task

Given a layer table, count the trainable parameters, total parameters, and their ratio under the BitFit convention.

- Input: \(\texttt{layer\_spec}\in\mathbb Z^{M\times 3}\) (int64), each row \([\text{kind},a,b]\): kind 0 = Linear(\(a\to b\), bias=True), kind 1 = LayerNorm(\(a\)) (\(b=0\) then), kind 2 = Linear(\(a\to b\), bias=False).
- Output \((\texttt{n\_trainable},\ \texttt{n\_total},\ \texttt{ratio})\): the first two are integers, \(\texttt{ratio}=\texttt{n\_trainable}/\texttt{n\_total}\) is a float.

**Convention**: \(M\ge 1\), \(1\le a,b\); \(b\) is ignored when kind=1; ratio is expressed exactly with float division. Tolerance \(10^{-5}\).

## Hints

- LayerNorm's \(\gamma\) is a weight, not a bias — BitFit does not train it (the most commonly tested trap).
- A Linear's bias length is the **output** dimension \(b\), not \(a\).
- When no layer has a bias, the ratio is 0 (total is still positive — don't crash the division).
