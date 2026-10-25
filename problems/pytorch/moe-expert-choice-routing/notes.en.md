# Solution Notes · Expert-Choice Routing

## Key insight: transpose the direction

- Conventional top-k: each token picks $k$ experts → each token gets exactly $k$ assignments, but the load per expert is uneven.
- EC: each expert picks $k_c$ tokens → each expert gets exactly $k_c$ load, but the number of assignments per token varies (0/1/2/…).

EC's capacity allocation is **inherently perfectly balanced**, requiring no aux loss. The cost: some tokens are "homeless" (picked by no expert) and need a fallback.

## tie-break

`torch.topk(probs[:, e], k_c)` does not guarantee the order of tokens with equal probs; the test cases avoid strict ties. If you need tie-break by ascending token index, use `sorted(range(T), key=lambda t: (-probs[t,e], t))[:k_c]`.

## renorm denominator

`weights.sum(-1, keepdim=True)` may be 0 (a token selected in no column). Use `torch.where(sum>0, w/sum, 0)` to zero it explicitly, avoiding `nan = 0/0`.

## Exactly $k_c$ per column

This is EC's core guarantee: `mask[:, e].sum() == k_c` for all e. The tests assert this invariant.

## Total activated parameters

If $E \cdot k_c = T \cdot k$, then on average each token is picked by $k$ experts — but **how many pick each individual token is not fixed**: some get 0, some get more than $k$.
