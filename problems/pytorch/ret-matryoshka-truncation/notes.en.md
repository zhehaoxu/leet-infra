# Solution Notes · Matryoshka Embedding Truncation

## Source and conventions

The MRL paper (arXiv:2205.13147) is about the training-side multi-scale loss (each prefix of the
embedding goes through its own InfoNCE); the inference-side usage (matryoshka adapters, OpenAI
text-embedding-3's `dimensions` parameter) uniformly does "**take the first d dims → re-run L2
normalization → cosine similarity**". This problem judges exactly that inference ledger and stays
away from training.

Two conventions that must be pinned:

- **Re-normalization**: after truncation the vector norm is no longer 1; a raw dot product mixes
  magnitude information back in and produces a ranking different from cosine. The
  `no-trunc-renorm` / `dot-no-norm` mutations exist to pin this down.
- **Zero vectors**: a row may become all-zero at some truncation level (orthogonal to the query
  there), and normalization would divide by zero. Pinned as `norm = max(‖v‖₂, 1e-12)` — matching
  `F.normalize`'s eps semantics; a zero vector's similarity is 0 and it joins tie ordering.

## Ledger design trade-offs

- `overlap_full` uses **set** intersection rather than positional alignment (the count of
  `top_d[i] == top_full[i]`): MRL cares about "can the coarse pass recall the fine pass's
  candidates" (recall@k semantics), and small positional shifts are not distortion. The
  `overlap-order` mutation checks whether the implementer really understands this.
- `truncated` is a dict keyed by `d`, decoupling the ledger from the `dims` order (`dims` may be
  unordered and repeated), so "processing order" never becomes observable behavior.
- The `top` sort key `(-sim, index)` matches the repo's other top-k problems: ties break to the
  smaller index.

## Two traps in the hand-computed anchor

1. **One-dimensional truncation leaves only the sign**: `normalize([x])` destroys all magnitude
   information, so similarities are exactly ±1 or 0 (zero vector). The anchor's d=1 top is the
   tie-broken `[0, 1, 3]`, not the magnitude order — the first hand computation sorted by
   magnitude and was wrong.
2. **Don't trust recalled angle tables for cos**: the cosine of `doc1=[1,2]` against `q=[1,1]` is
   `(1+2)/(√2·√5)=3/√10≈0.949`; a memorized "cos(18°)≈0.951" leaks a small mismatch (the angle is
   really 18.43°). Anchor assertions always use closed-form expressions, never decimal approximations.

## Testing and mutations

- The oracle is pure Python (math + lists), cross-validated on 30 random cases (float64
  element-wise cosine, matching torch's double path, tolerance 1e-6).
- The rank-preservation-spectrum case builds embeddings where the first 2 dims are the principal
  components and the last 2 dims carry tiny noise, asserting the d=2 top equals the full top —
  exactly the property MRL buys.
- All 8 mutations caught: no re-normalization after truncation / query not normalized / tail d
  dims / no normalization anywhere (dot product) / ascending sort / ties to larger index /
  overlap as positional agreement / silent clamp of out-of-range top_k.
