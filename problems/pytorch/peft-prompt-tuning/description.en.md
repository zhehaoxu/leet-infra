# PEFT: Prompt-Tuning Soft-Prompt Embedding Concatenation

## Problem

**Prompt-tuning** (Lester et al., EMNLP 2021, arXiv:2104.08691) is the minimal version of prefix-tuning: it only prepends \(L\) learnable soft-prompt tokens at the **input embedding layer**, and they enter the model together with the embeddings. On T5-XXL it matches full fine-tuning with \(L\cdot d\) parameters per task (the larger the model, the smaller the gap), and multiple tasks share one frozen model — each task stores only a tiny prompt table.

Forward: given input embeddings \(E\in\mathbb R^{S_0\times d}\), task \(t\)'s soft prompt is fetched from a shared pool (the multi-task serving form):

\[X_t=\bigl[\text{pool}[\text{task\_ids}[t]];\ E\bigr]\in\mathbb R^{(L+S_0)\times d}\]

i.e. first gather the task's \(L\) prompt rows by row index, then concatenate the original embeddings with the **prefix in front**. The trainable parameter count per task is \(n_\text{params}=T\cdot L\cdot d\) (for \(T\) tasks) — independent of the pool size \(P\), which is only a storage form.

## Task

Given the soft-prompt pool, the per-task row indices, and the input embeddings, produce the concatenated batched embeddings and the parameter count.

- Inputs: \(\texttt{pool}\in\mathbb R^{P\times d}\), \(\texttt{task\_ids}\in\mathbb Z^{T\times L}\) (each row holds \(L\) distinct pool row indices, \(0\le\text{idx}<P\)), \(E\in\mathbb R^{S_0\times d}\).
- Output \((X,\ \texttt{n\_params})\): \(X\in\mathbb R^{T\times(L+S_0)\times d}\) (float32, \(X[t,:L]=\text{pool}[\text{task\_ids}[t]]\), \(X[t,L:]=E\)), \(\texttt{n\_params}=T\cdot L\cdot d\).

**Convention**: compute in float64 internally, output float32; no ready-made PEFT/transformers library. Tolerance \(10^{-5}\).

## Hints

- Gather is a per-task row-index operation: `task_ids` is 2D; row \(t\) holds the \(L\) pool rows task \(t\) wants.
- Concatenation order: prompt in front, original embeddings behind.
- \(n_\text{params}\) counts only the "per-task exclusive view" of trainable parameters, \(T\cdot L\cdot d\), independent of \(P\).
