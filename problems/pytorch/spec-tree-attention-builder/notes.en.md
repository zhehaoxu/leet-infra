# Solution Notes · Candidate Tree Construction

## Why a heap suffices

Naively, "pick the best N of all \(\sum_d K^d\) paths" sounds like full enumeration plus sorting. But candidate logprobs are non-positive, so **accumulating along a path can only make the score worse**; a parent is never worse than its children. Given that monotonicity, best-first pop order equals global order, and the heap holds at most "popped nodes \(\times K\)" paths, so enumeration is unnecessary.

Now prefix closure: a child is queued only after its parent is popped, so every emitted node already has an emitted ancestor with an assigned index—`parent` is always valid and smaller. That is also the premise for filling `mask` directly by index.

## How the three structures relate

- `parent` / `depth` describe the **topology**: who is whose parent and at which level.
- `positions` describes **positional semantics**: \(prefix\_len + depth - 1\), shared by all nodes at a depth. This enables KV reuse across branches at the same position and makes RoPE relative distances match serial decoding exactly.
- `mask` describes **visibility**: the prefix is always visible, and node columns contain only self plus ancestors. Mutual invisibility between siblings is essential—otherwise the suffix of branch A would attend to tokens of branch B and every verified logit would be wrong.

They are independent yet mutually constraining: nodes at one depth share a `position` but cannot see each other (`positions` equal, `mask` entries 0). That is the most counterintuitive part of tree attention.

## Ties and budget

- The tie key is the lexicographic order of the whole path's candidate indices—not the rank inside a head, and not the token id: paths are the heap objects, so comparing paths keeps the order stable and reproducible.
- `budget` maps directly to the sequence length of the tree forward pass and is the main memory/latency knob; budget=1 degenerates to betting on a single token, while a generous budget exhausts the whole tree.
- The tree's maximum depth is the number of heads \(H\), so the node count is bounded by \(\sum_{d=1}^{H} K^d\); extra budget yields no additional nodes.

## Complexity

Heap operations cost \(O(N\log(NK))\) and mask filling \(O(N\cdot H)\), so overall it is on the order of \(O(NH)\) with \(N\) nodes. In real systems the construction runs once on the CPU and is reused across decoding steps, so the cost is not the bottleneck here—this problem is about structural correctness.

## Related problems

- `spec-medusa-tree-verify`: performs typical acceptance verification on an **already assembled path set**; this problem supplies the missing "how the tree is built".
- `spec-eagle2-dynamic-tree-prune`: EAGLE-2 prunes dynamically by draft confidence—the adaptive version of this fixed-budget best-first scheme.
- `spec-specinfer-token-tree-verify`: verification-side logic once the tree is flattened into a sequence; upstream/downstream of this problem's mask generation.
