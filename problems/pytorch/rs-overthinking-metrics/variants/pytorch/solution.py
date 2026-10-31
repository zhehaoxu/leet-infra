
def solve(correct, tokens, bin_edges):
    for x in correct:
        if not isinstance(x, bool):
            raise ValueError("correct must be a list of bools")
    for t in tokens:
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise ValueError("tokens must be non-negative ints")
    if not isinstance(bin_edges, (list, tuple)) or len(bin_edges) < 2:
        raise ValueError("bin_edges must have at least 2 edges")
    prev = None
    for e in bin_edges:
        if isinstance(e, bool) or not isinstance(e, int):
            raise ValueError("bin_edges must be ints")
        if prev is not None and e <= prev:
            raise ValueError("bin_edges must be strictly increasing")
        prev = e
    if len(correct) != len(tokens):
        raise ValueError("correct/tokens length mismatch")
    n = len(correct)
    if n == 0:
        raise ValueError("empty records")

    k = len(bin_edges) - 1
    counts = [0] * k
    correct_counts = [0] * k
    token_sums = [0] * k
    for c, t in zip(correct, tokens):
        j = k - 1
        for b in range(k - 1):
            if bin_edges[b] <= t < bin_edges[b + 1]:
                j = b
                break
        counts[j] += 1
        token_sums[j] += t
        if c:
            correct_counts[j] += 1

    bins = []
    for j in range(k):
        acc = (correct_counts[j] / counts[j]) if counts[j] > 0 else None
        mt = (token_sums[j] / counts[j]) if counts[j] > 0 else None
        bins.append({"count": counts[j], "accuracy": acc, "mean_tokens": mt})

    n_correct = sum(1 for c in correct if c)
    sum_correct = sum(t for c, t in zip(correct, tokens) if c)
    sum_wrong = sum(t for c, t in zip(correct, tokens) if not c)
    len_correct = (sum_correct / n_correct) if n_correct > 0 else None
    n_wrong = n - n_correct
    len_wrong = (sum_wrong / n_wrong) if n_wrong > 0 else None
    efficiency_ratio = (len_wrong / len_correct) if (len_correct and len_wrong is not None) else None

    best_bin = None
    best_acc = -1.0
    for j in range(k):
        if counts[j] > 0 and bins[j]["accuracy"] > best_acc:
            best_acc = bins[j]["accuracy"]
            best_bin = j

    return {
        "n": n,
        "accuracy": n_correct / n,
        "total_tokens": sum(tokens),
        "bins": bins,
        "len_correct": len_correct,
        "len_wrong": len_wrong,
        "efficiency_ratio": efficiency_ratio,
        "best_bin": best_bin,
    }
