# Auto-CoT Cluster-Representative Demo Assembly

## Problem

The quality of few-shot CoT hinges on the demos. Auto-CoT (arXiv:2210.03493) builds a broad-coverage demo set automatically via **clustering**: pool the questions into k semantic clusters and pick **one** representative reasoning chain per cluster — less manual curation, automatic coverage of the question space.

You are given (encoding and clustering happen outside the pool; `retrieval-kmeans-lloyd` already covers the k-means operator, so the assignment and distances are explicit inputs):

- `questions` / `rationales` / `pred_anses`: the candidate pool (raw question, zero-shot-CoT rationale, predicted answer), equal-length list[str];
- `cluster_assignment`: each sample's cluster id (the official KMeans.labels_);
- `dist`: Euclidean distances from each sample to each cluster centroid, (n, k) (the official clustering_model.transform);
- `max_ra_len`: rationale line-count cap (default 5; the official last_letters tasks pass 7);
- `arithmetic`: whether the arithmetic answer verification applies;
- `gold_anses`: optional gold answers (collected only on the official debug path).

## The pipeline to implement

### 1. Cluster

Cluster i collects the samples with assignment==i, appended in ascending index order, recording the distance to centroid i and the global index.

### 2. Intra-cluster ordering

Sort by distance **stably ascending** (the official sorted is stable, so distance ties keep the global-index order).

### 3. Filter candidates one by one (any failure moves to the next candidate)

```
a. question words   len(question.strip().split()) <= 60
b. rationale lines  len(r.replace("\n\n", "\n").split("\n")) <= max_ra_len
                    # r = rationale.strip()
c. ends with period r.endswith(".")            # official r[-1]=="."; empty strings fail
d. non-empty answer pred_ans.strip() != ""
e. arithmetic check when arithmetic: pred_ans must appear in
              r.split(".")[-2] (the penultimate sentence; condition c guarantees at least two segments)
              or r.split()[-10:] (the last 10 words); otherwise skip
```

### 4. At most one per cluster

The first candidate passing every filter becomes the cluster's demo and `break`s; if every candidate is filtered out (or the cluster is empty) the cluster contributes no demo. Clusters are processed in id order 0..k-1.

### 5. Demo assembly

```
rationale_clean = " ".join(r.replace("\n\n", "\n").replace("\n", " ").strip().split())
                # the official cleaning chain; split() folds all whitespace to single spaces
question / pred_ans pass through unchanged (the official "\nA:" suffix and "A: Let's think..." replacement are log-parsing layer)
gold_ans = gold_anses[sid].strip() (when given) or None
```

Return `dict(demos, selected, cluster_order, num_clusters)`: the demo list, the global indices of the selected samples, each demo's source cluster id, and the cluster count k.

## Input constraints

- `questions`/`rationales`/`pred_anses` are equal-length non-empty list[str]; `cluster_assignment` is an equal-length list[int] (bools do not count) with values in [0, k); `dist` is an (n, k) list[list[float]] (bools do not count), k ≥ 1;
- `max_ra_len` is an integer ≥ 1 (bools do not count); `arithmetic` is a bool; `gold_anses`, when given, is an equal-length list[str].

## Hints

- `r.split(".")[-2]` in filter e is only safe after condition c — c guarantees at least one period, so the split has at least two segments;
- The empty-preferred-heap fallthrough, the `random.shuffle` sampling (sampling≠center), and the log-file parsing (the Q:/A:/pred_mode/GT : line protocol) are outside the judging scope;
- The columns of the distance matrix for **clusters other than the assigned one are dead data** (the official code only reads `dist[sentence_id][cluster_id]`), but reading the wrong column changes the intra-cluster ordering.
