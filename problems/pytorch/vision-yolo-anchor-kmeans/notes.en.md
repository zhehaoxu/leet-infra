# Notes: vision-yolo-anchor-kmeans

## Source

AlexeyAB/darknet `scripts/kmeansiou.c` (all 391 lines): `closest_center`/`kmeans_expectation`/`kmeans_maximization`/`WCSS`/`smart_centers`/`do_kmeans` verified one by one. `dist()` computes `inter = min·min` and `return 1 - iou`. Original pjreddie darknet has no k-means code (the paper author ran it offline); AlexeyAB's is the community standard.

## Key Judgments

- **The empty-cluster quirk was an accidental find**: `kmeans_maximization` does `zero_matrix(centers)` then divides only `if(counts[i])` — an empty cluster's centroid is literally (0,0), not resampled or dropped. The first test draft tried k=2 init=[0,0] to trigger it, but assignments start at calloc-0 and the first E converges, so the quirk never executes; k=3 init=[0,1,1] actually triggers it (M step leaves cluster2 empty).
- **Why the k=1 special case exists**: all-0 assignments + everything closest to 0 → expectation always converges → no M step → centroid stuck at the random init. The official `if(k==1) kmeans_maximization(...)` patches exactly this. Mutation `k1-no-special` pins it.
- **WCSS is a misnomer**: `sum += (1 - dist(...))` is mean IoU — transcribe the code, not the name.
- **smart_centers (D² sampling) out of scope**: depends on the rand() stream; replaced by explicit init indices, declared in notes.

## Boundary

- No tries/restarts (random stream), no k-means++ D² sampling itself, no `-c` count output.
- Bounded against `vision-anchor-match`: inference-time anchor-GT matching vs pre-training clustering.
- Mutation lesson: `inter-max` (min→max) turns IoU into values > 1, breaking Lloyd's monotone convergence → official has no max_iter → **the harness hangs forever**. Replaced with the bounded order-preserving scaling `*0.5` (caught via anchor avg_iou). Always check convergence before writing distance-breaking mutations.

## Test Design

- Oracle `_py` replays Lloyd independently (E/M rewritten).
- 9 checks: two-cluster anchor (hand-computed avg_iou fractions), quirk not reached, quirk triggered, k=1 special case, tie-to-lower, iou boundary, 150 random, 6 invalid tiers.
- Mutations 9/9: inter scaled, union without subtraction, tie-to-larger, empty cluster (1,1), k=1 special removed, mean +1, WCSS denominator +1, init w/h swapped, convergence check short-circuited.
