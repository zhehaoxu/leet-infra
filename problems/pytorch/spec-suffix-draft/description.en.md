# Suffix / REST: Longest-Suffix Retrieval Draft

## Problem

Speculative decoding without a trained draft model: **retrieval-based drafts** simply "copy" a continuation from previously generated text/a corpus. **REST** (Zhao et al., 2023, [arXiv:2311.08252](https://arxiv.org/abs/2311.08252)) searches an offline-built **datastore** for a suffix matching the current context and uses the tokens following it as the draft; **Suffix Decoding** (Arora et al., 2024, [arXiv:2411.04975](https://arxiv.org/abs/2411.04975)) uses a suffix tree to make this "longest-suffix match" online and dynamically deepened. Both share one core operation: **find the longest suffix of context that occurs as a contiguous substring of some datastore record; the draft is the next few tokens after the match position**. This problem implements that retrieval operation.

## Rules

Arguments:
- `datastore`: a list of token sequences `list[list[int]]`, each a record (entry).
- `context`: the current context `list[int]` (including its last few tokens).
- `gamma`: maximum number of draft tokens.
- `min_match`: minimum length threshold for a suffix match; anything shorter does not count as a match.

Search the datastore for a **suffix** of context $s = (\text{context}_{-n},\dots,\text{context}_{-1})$ ($n\ge \text{min\_match}$) such that $s$ occurs as a **contiguous substring** of some entry at position $i$ (i.e. $\text{entry}[i:i+n]=s$). The draft is $[\text{entry}_{i+n},\dots,\text{entry}_{i+n+\gamma-1}]$ (truncated if it runs past the end).

**Selection and tie-break** (guaranteeing uniqueness):
1. Prefer the **longest** suffix match (largest $n$);
2. when $n$ ties, take the **smallest entry index** (the earlier entry);
3. when both the entry and $n$ tie, take the **smallest match position $i$** (the earliest occurrence within that entry).

If no suffix match satisfies `min_match`, return an **empty** draft. Return a one-dimensional `torch.long` tensor.

- Unlike Lookahead (problem 179), which uses a fixed-length window and falls back step by step, this is a **variable-length longest suffix plus a one-shot grab of at most gamma draft tokens**, with no per-token fallback.
- "Longest suffix first" matches the intuition: the longer the match, the more trustworthy the copied continuation.

## Examples

```
datastore = [[1,2,3,4,5],
             [9,2,3,7,8],
             [2,3,6]]
context=[0,1,2,3], gamma=2, min_match=2

Suffixes from longest to shortest:
 n=4 [0,1,2,3] -> none
 n=3 [1,2,3]   -> hits entry0 at i=0, draft entry0[3:5] = [4,5]   (longest, locked in)
 n=2 [2,3]     -> a longer match is already locked, ignored
Output = [4, 5]
```

## Constraints

- CPU; return a one-dimensional `torch.long` (possibly empty), compared with `torch.equal`.
- Draft length $\le \gamma$, and never beyond the remaining length of the match position inside the entry.
- The three tie-break priorities follow the statement strictly: longer → earlier entry → earlier position.
