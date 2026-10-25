# Solution Notes · ObservedAttention

## Difference from H2O

H2O uses the **accumulated attention** (the total attention each key has received from its generation up to the current step); kvpress's ObservedAttention divides by `L - j` (the number of queries that can see it) to get an **average** — avoiding early keys from scoring naturally high just because they "survived longer".

## The Actual Shape of the Three Steps

```python
scores = attn.sum(2)                     # (B, Hq, K) — sum over queries
n_in_sum = torch.arange(L, 0, -1)         # each key is seen by (L - j) queries
scores = scores / n_in_sum                # average
scores = scores.view(B, H_kv, groups, L).mean(2)  # kv-group average
```

## kv-group Averaging

In GQA, multiple Q-heads share one KV-head. `attn` is a `num_heads` tensor while `keys` is a `num_key_value_heads` tensor. To score the KV heads you must **average the scores of the Q-heads within each group**. In this problem `attn` is given directly at `Hq`, with `groups = Hq // H`.

## Why the Denominator Is L - j

Position $j$ is the key (column), and only queries (rows) with $i \geq j$ can see it — under the causal mask, entries with $i < j$ are 0. So summing over attn and then dividing by the "effective query count" yields the average.
