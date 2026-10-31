"""Independent pure-Python PruMerge oracle: manual dot products, softmax and clusters."""

import math

import torch


def _keep_count(tokens, reduction_ratio):
    return int(tokens * float(reduction_ratio))


def _oracle(image_features, cls_attn, reduction_ratio, eps=1e-12):
    """Cosines, softmax and the residual merge from scalar loops; no torch selection."""
    features = image_features.tolist()
    attention = cls_attn.tolist()
    batch = len(features)
    tokens = len(features[0])
    channels = len(features[0][0])
    keep = _keep_count(tokens, reduction_ratio)
    scale = channels ** -0.5

    merged_rows, kept_rows, pruned_rows = [], [], []
    cluster_rows, weight_rows, score_rows = [], [], []
    for row in range(batch):
        ranked = sorted(range(tokens), key=lambda index: (-attention[row][index], index))
        kept = sorted(ranked[:keep])
        chosen = set(kept)
        pruned = [index for index in range(tokens) if index not in chosen]

        def unit(index):
            vector = features[row][index]
            norm = math.sqrt(sum(component * component for component in vector))
            denominator = norm if norm > eps else eps  # max(norm, eps) floor
            return [component / denominator for component in vector]

        kept_units = [unit(index) for index in kept]
        assigned = [[] for _ in kept]
        clusters, weights, best_similarities = [], [], []
        for position, index in enumerate(pruned):
            candidate = unit(index)
            similarity = [
                sum(a * b for a, b in zip(candidate, kept_units[column]))
                for column in range(keep)
            ]
            best = 0
            for column in range(1, keep):
                if similarity[column] > similarity[best]:
                    best = column  # strict comparison keeps the FIRST maximal column
            peak = max(similarity)
            exponentials = [math.exp(value - peak) for value in similarity]
            probability = [value / sum(exponentials) for value in exponentials]
            clusters.append(best)
            best_similarities.append(similarity[best])
            weights.append(probability[best] * scale)
            assigned[best].append(position)

        merged = []
        for column in range(keep):
            accumulated = list(features[row][kept[column]])
            for position in assigned[column]:
                weight = weights[position]
                vector = features[row][pruned[position]]
                accumulated = [total + weight * part
                               for total, part in zip(accumulated, vector)]
            merged.append(accumulated)
        merged_rows.append(merged)
        kept_rows.append(kept)
        pruned_rows.append(pruned)
        cluster_rows.append(clusters)
        weight_rows.append(weights)
        score_rows.append(best_similarities)

    dtype = image_features.dtype
    return (
        torch.tensor(merged_rows, dtype=dtype),
        torch.tensor(kept_rows, dtype=torch.int64),
        torch.tensor(pruned_rows, dtype=torch.int64),
        torch.tensor(cluster_rows, dtype=torch.int64),
        torch.tensor(weight_rows, dtype=dtype),
        torch.tensor(score_rows, dtype=dtype),
    )


def _case(b=2, n=9, c=4, ratio=0.5, dtype=torch.float64, seed=11, eps=1e-12, coarse=True):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn((b, n, c), dtype=dtype, generator=generator) * 0.8
    attention = torch.rand((b, n), dtype=dtype, generator=generator)
    if coarse:  # exact ties so both paths must apply the same tie rules
        attention = (attention * 4).round() / 4
    return (features, attention, ratio, eps)


def _example():
    """Kept 1 and 3; three pruned tokens collapse onto kept column 0."""
    features = torch.tensor(
        [[[1.0, 1.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, -1.0]]],
        dtype=torch.float64,
    )
    attention = torch.tensor([[0.2, 0.9, 0.4, 0.7, 0.1]], dtype=torch.float64)
    return (features, attention, 0.5, 1e-12)


def _zero_norm_example():
    features = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
          [2.0, 2.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float64,
    )
    attention = torch.tensor([[0.6, 0.6, 0.9, 0.2]], dtype=torch.float64)
    return (features, attention, 0.25, 1e-12)


def _close(actual, expected):
    tolerance = 4e-5 if expected.dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _verify(solve, args):
    image_features, cls_attn, reduction_ratio = args[0], args[1], args[2]
    eps = args[3] if len(args) > 3 else 1e-12
    expected = _oracle(image_features, cls_attn, reduction_ratio, eps)
    result = solve(image_features, cls_attn, reduction_ratio, eps)
    assert isinstance(result, tuple) and len(result) == 6
    batch, tokens, channels = image_features.shape
    keep = _keep_count(tokens, reduction_ratio)
    for actual, target in zip(result, expected):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == target.shape and actual.dtype == target.dtype
        assert actual.device == image_features.device
        assert torch.isfinite(actual).all()
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = result
    assert merged.shape == (batch, keep, channels)
    assert kept_indexs.shape == (batch, keep)
    assert pruned_indexs.shape == (batch, tokens - keep)
    assert cluster.shape == (batch, tokens - keep)
    assert weights.shape == scores.shape == (batch, tokens - keep)
    assert kept_indexs.dtype == torch.int64 and weights.dtype == image_features.dtype
    assert torch.equal(kept_indexs, expected[1]) and torch.equal(pruned_indexs, expected[2])
    assert torch.equal(cluster, expected[3])
    _close(merged, expected[0])
    _close(weights, expected[4])
    _close(scores, expected[5])
    assert (kept_indexs[:, 1:] > kept_indexs[:, :-1]).all(), "kept must ascend"
    assert (pruned_indexs[:, 1:] > pruned_indexs[:, :-1]).all(), "pruned must ascend"
    for row in range(batch):
        assert sorted(kept_indexs[row].tolist() + pruned_indexs[row].tolist()) == list(range(tokens))
        if tokens > keep:
            threshold = max(cls_attn[row, kept_indexs[row]].tolist())
            assert threshold >= max(cls_attn[row, pruned_indexs[row]].tolist())
            assert cluster[row].min().item() >= 0 and cluster[row].max().item() < keep
            # a kept token with no assignment is returned verbatim
            for column in set(range(keep)) - set(cluster[row].tolist()):
                assert torch.equal(merged[row, column], image_features[row, kept_indexs[row, column]])
    return result


def check_example_two_kept_patches(solve):
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(solve, _example())
    assert kept_indexs.tolist() == [[1, 3]] and pruned_indexs.tolist() == [[0, 2, 4]]
    assert cluster.tolist() == [[0, 0, 0]]
    root = math.sqrt(0.5)
    _close(scores, torch.tensor([[root, 1.0, root]], dtype=torch.float64))
    _close(weights, torch.tensor(
        [[0.5 * root,
          math.exp(1.0) / (math.exp(1.0) + 1.0) * root,
          math.exp(root) / (math.exp(root) + math.exp(-root)) * root]],
        dtype=torch.float64))
    _close(merged[0, 1], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert weights.sum().item() > 1.0, "cluster weights must not renormalise to 1"
    _close(merged[0, 0], torch.tensor(
        [1.0 + weights[0, 0].item() + 2.0 * weights[0, 1].item() + weights[0, 2].item(),
         weights[0, 0].item() - weights[0, 2].item()], dtype=torch.float64))


def check_example_zero_norm_single_keep(solve):
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
        solve, _zero_norm_example())
    assert kept_indexs.tolist() == [[2]] and pruned_indexs.tolist() == [[0, 1, 3]]
    assert cluster.tolist() == [[0, 0, 0]]
    _close(weights, torch.full((1, 3), 0.5, dtype=torch.float64))  # C = 4 -> C ** -0.5 = 0.5
    _close(scores, torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64))
    _close(merged, torch.tensor([[[3.0, 2.5, 0.5, 0.0]]], dtype=torch.float64))


def check_truncation_not_rounding(solve):
    table = ((5, 0.5, 2), (4, 0.25, 1), (3, 0.5, 1), (7, 0.5, 3), (9, 0.6, 5), (3, 0.99, 2))
    for tokens, ratio, keep in table:
        assert _keep_count(tokens, ratio) == keep, (tokens, ratio)
        args = _case(b=1, n=tokens, c=3, ratio=ratio, seed=tokens + int(100 * ratio))
        merged, kept_indexs, pruned_indexs, _, _, _ = _verify(solve, args)
        assert merged.shape[1] == keep and kept_indexs.shape[1] == keep
        assert pruned_indexs.shape[1] == tokens - keep


def check_all_attention_tied_keeps_lowest(solve):
    generator = torch.Generator().manual_seed(3)
    features = torch.randn((2, 8, 3), dtype=torch.float64, generator=generator)
    attention = torch.full((2, 8), 0.5, dtype=torch.float64)
    merged, kept_indexs, pruned_indexs, cluster, weights, _ = _verify(
        solve, (features, attention, 0.5))
    assert kept_indexs.tolist() == [[0, 1, 2, 3]] * 2
    assert pruned_indexs.tolist() == [[4, 5, 6, 7]] * 2
    assert merged.shape == (2, 4, 3) and cluster.shape == (2, 4)
    assert (cluster < 4).all() and weights.shape == (2, 4)
    assert torch.equal(kept_indexs, torch.arange(4).expand(2, 4))


def check_full_ratio_keeps_every_patch(solve):
    for args in (_case(b=2, n=6, c=2, ratio=1.0), _case(b=1, n=5, c=4, ratio=1.0, seed=8),
                 (_zero_norm_example()[0], _zero_norm_example()[1], 1.0, 1e-12)):
        merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(solve, args)
        assert torch.equal(kept_indexs, torch.arange(args[0].shape[1]).expand(kept_indexs.shape))
        assert pruned_indexs.shape == (args[0].shape[0], 0)
        assert cluster.shape == (args[0].shape[0], 0) and weights.numel() == 0
        assert scores.numel() == 0
        assert torch.equal(merged, args[0]), "no pruned token means no merge at all"


def check_cluster_of_two_does_not_renormalise(solve):
    # Two pruned tokens collapse onto kept column 0. Their weights are the softmax
    # probabilities taken over the WHOLE kept set times C ** -0.5, so the cluster
    # sums to 2 * e / (e + 1) * 2 ** -0.5 instead of the 1.0 an in-cluster
    # renormalisation would produce.
    features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [2.0, 0.0]]], dtype=torch.float64
    )
    attention = torch.tensor([[0.9, 0.8, 0.3, 0.2]], dtype=torch.float64)
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
        solve, (features, attention, 0.5))
    assert kept_indexs.tolist() == [[0, 1]] and pruned_indexs.tolist() == [[2, 3]]
    assert cluster.tolist() == [[0, 0]]
    _close(scores, torch.full((1, 2), 1.0, dtype=torch.float64))
    probability = math.exp(1.0) / (math.exp(1.0) + 1.0)
    _close(weights, torch.full((1, 2), probability * math.sqrt(0.5), dtype=torch.float64))
    total = weights.sum().item()
    assert abs(total - 1.0) > 1e-6, "in-cluster renormalisation is the wrong rule"
    _close(merged[0, 0], torch.tensor([1.0 + 5.0 * probability * math.sqrt(0.5), 0.0],
                                      dtype=torch.float64))
    assert torch.equal(merged[0, 1], features[0, 1]), "empty cluster stays verbatim"


def check_channel_scaling_after_softmax(solve):
    # C ** -0.5 uses the feature dimension and is applied AFTER the softmax.
    for channels, expected in ((1, 1.0), (4, 0.5), (9, 1 / 3)):
        features = torch.zeros((1, 3, channels), dtype=torch.float64)
        features[0, 0, 0] = 2.0   # kept
        features[0, 1, 0] = 1.0   # pruned, same direction
        if channels > 1:  # a one-wide feature cannot be orthogonal, so it stays zero
            features[0, 2, 1] = 1.0  # pruned, cosine 0 with the kept patch
        attention = torch.tensor([[0.9, 0.5, 0.2]], dtype=torch.float64)
        merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
            solve, (features, attention, 1 / 3))
        assert kept_indexs.tolist() == [[0]] and pruned_indexs.tolist() == [[1, 2]]
        assert cluster.tolist() == [[0, 0]]
        _close(scores, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
        # a single kept candidate makes the softmax probability exactly 1
        _close(weights, torch.full((1, 2), expected, dtype=torch.float64))
        _close(merged[0, 0, 0], torch.tensor(2.0 + expected, dtype=torch.float64))
    # two kept candidates: each probability comes from the softmax over the WHOLE
    # kept set, then is gathered at the chosen column and scaled by C ** -0.5
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]],
                            dtype=torch.float64)
    attention = torch.tensor([[0.9, 0.8, 0.1, 0.2]], dtype=torch.float64)
    _, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
        solve, (features, attention, 0.5))
    assert kept_indexs.tolist() == [[0, 1]] and pruned_indexs.tolist() == [[2, 3]]
    assert cluster.tolist() == [[1, 0]], "the tied second row must take column 0"
    root = math.sqrt(0.5)
    _close(scores, torch.tensor([[1.0, root]], dtype=torch.float64))
    _close(weights, torch.tensor(
        [[math.exp(1.0) / (math.exp(1.0) + 1.0) * root, 0.5 * root]], dtype=torch.float64))


def check_zero_norm_patch_has_no_nan(solve):
    features = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 4.0]]],
        dtype=torch.float64,
    )
    attention = torch.tensor([[0.9, 0.8, 0.7, 0.6]], dtype=torch.float64)
    reference = None
    for eps in (1e-30, 1e-12, 1.0):
        result = _verify(solve, (features, attention, 0.5, eps))
        if reference is not None:
            for first, second in zip(result, reference):
                if first.dtype == torch.int64:
                    assert torch.equal(first, second)
                else:
                    _close(first, second)
        reference = result
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = reference
    assert kept_indexs.tolist() == [[0, 1]] and pruned_indexs.tolist() == [[2, 3]]
    assert cluster.tolist() == [[0, 1]], "the all-zero cosine row falls to the first column"
    assert torch.isfinite(merged).all() and torch.isfinite(weights).all()
    _close(merged[0, 0], torch.zeros(3, dtype=torch.float64))
    # the kept zero patch has cosine 0 with everything, so index 3 scores <11/15|11/15>
    cosine = 11.0 / 15.0
    probability = math.exp(cosine) / (1.0 + math.exp(cosine))
    root = 3 ** -0.5
    _close(scores, torch.tensor([[0.0, cosine]], dtype=torch.float64))
    _close(weights, torch.tensor([[0.5 * root, probability * root]], dtype=torch.float64))
    _close(merged[0, 1], torch.tensor([1.0 + 3.0 * probability * root, 2.0,
                                       2.0 + 4.0 * probability * root], dtype=torch.float64))


def check_argmax_tie_first_index(solve):
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]],
                            dtype=torch.float64)
    attention = torch.tensor([[0.4, 0.3, 0.9, 0.8]], dtype=torch.float64)
    # kept columns 2 and 3 are the same direction, so every pruned token is tied
    # between the two columns and must fall into the FIRST one
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
        solve, (features, attention, 0.5))
    assert kept_indexs.tolist() == [[2, 3]] and pruned_indexs.tolist() == [[0, 1]]
    assert cluster.tolist() == [[0, 0]], "equal cosines must take the FIRST column"
    _close(scores, torch.tensor([[1.0, 0.0]], dtype=torch.float64))
    _close(weights, torch.full((1, 2), 0.5 * math.sqrt(0.5), dtype=torch.float64))
    _close(merged[0, 0], torch.tensor([1.0 + 0.5 * math.sqrt(0.5), 0.5 * math.sqrt(0.5)],
                                      dtype=torch.float64))
    assert torch.equal(merged[0, 1], features[0, 3]), "unassigned kept column is verbatim"


def check_empty_cluster_returned_verbatim(solve):
    features = torch.tensor(
        [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [5.0, 0.0]]], dtype=torch.float64
    )
    attention = torch.tensor([[0.9, 0.8, 0.1, 0.2]], dtype=torch.float64)
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(
        solve, (features, attention, 0.5))
    assert kept_indexs.tolist() == [[0, 1]] and pruned_indexs.tolist() == [[2, 3]]
    assert cluster.tolist() == [[0, 0]], "both pruned tokens collapse onto column 0"
    assert torch.equal(merged[0, 1], features[0, 1]), "empty cluster must be byte-exact"
    assert torch.count_nonzero(cluster[0] == 1) == 0
    root = 0.5 * 2 ** -0.5  # softmax over two tied candidates is 0.5, then C ** -0.5
    _close(weights, torch.full((1, 2), root, dtype=torch.float64))
    _close(merged[0, 0], torch.tensor([1.0 + root + 5.0 * root, 1.0 + root],
                                      dtype=torch.float64))


def check_residual_is_own_plus_assigned(solve):
    args = _case(b=2, n=8, c=3, ratio=0.375, seed=29)
    merged, kept_indexs, pruned_indexs, cluster, weights, _ = _verify(solve, args)
    scale = args[0].shape[2] ** -0.5
    for row in range(2):
        accumulator = {column: args[0][row, kept_indexs[row, column]].tolist()
                       for column in range(kept_indexs.shape[1])}
        for position, target in enumerate(cluster[row].tolist()):
            weight = weights[row, position].item()
            assert 0.0 < weight <= scale + 1e-12, "softmax probability times C ** -0.5"
            source = args[0][row, pruned_indexs[row, position]].tolist()
            accumulator[target] = [total + weight * part
                                   for total, part in zip(accumulator[target], source)]
        for column, values in accumulator.items():
            _close(merged[row, column], torch.tensor(values, dtype=args[0].dtype))


def check_scores_are_max_cosine(solve):
    args = _case(b=2, n=7, c=4, ratio=4 / 7, seed=13)
    merged, kept_indexs, pruned_indexs, cluster, weights, scores = _verify(solve, args)
    features = args[0]
    for row in range(2):
        for position, index in enumerate(pruned_indexs[row].tolist()):
            left = features[row, index]
            left_norm = max(torch.linalg.norm(left).item(), 1e-12)
            best = None
            for column, other in enumerate(kept_indexs[row].tolist()):
                right = features[row, other]
                cosine = torch.dot(left, right).item() / (
                    left_norm * max(torch.linalg.norm(right).item(), 1e-12))
                best = cosine if best is None or cosine > best else best
                if column == cluster[row, position].item():
                    assert abs(cosine - scores[row, position].item()) < 1e-9
            assert abs(scores[row, position].item() - best) < 1e-9
            assert abs(scores[row, position].item()) <= 1.0 + 1e-9
    # cosines ignore magnitude, so scaling one patch leaves scores and weights fixed
    scaled = args[0].clone()
    scaled[:, 0] = scaled[:, 0] * 3.0
    rescaled = _verify(solve, (scaled, args[1], args[2]))
    _close(rescaled[5], scores)
    _close(rescaled[4], weights)
    assert not torch.equal(rescaled[0], merged)


def check_float32_matches_float64(solve):
    args = _case(b=2, n=8, c=3, ratio=0.5, seed=41)
    wide = _verify(solve, args)
    narrow = _verify(solve, (args[0].to(torch.float32), args[1].to(torch.float32), 0.5))
    assert narrow[0].dtype == torch.float32 and narrow[4].dtype == torch.float32
    assert torch.equal(narrow[1], wide[1]) and torch.equal(narrow[2], wide[2])
    assert torch.equal(narrow[3], wide[3])
    for field in (0, 4, 5):
        torch.testing.assert_close(narrow[field].to(torch.float64), wide[field],
                                   rtol=1e-5, atol=1e-5)
    for dtype in (torch.float32, torch.float64):
        _verify(solve, _case(b=3, n=11, c=5, ratio=0.6, seed=19, dtype=dtype))


def check_noncontiguous_and_no_mutation(solve):
    for dtype in (torch.float32, torch.float64):
        features, attention, ratio, eps = _case(b=2, n=7, c=3, ratio=4 / 7, seed=23, dtype=dtype)
        dense = _verify(solve, (features, attention, ratio, eps))
        padded = torch.zeros((features.shape[0], features.shape[1], features.shape[2] * 2),
                             dtype=dtype)
        padded[..., ::2] = features
        padded[..., 1::2] = features.flip(-1) * 7.0
        stacked = torch.zeros((attention.shape[0], attention.shape[1] * 2), dtype=dtype)
        stacked[:, ::2] = attention
        stacked[:, 1::2] = 1.0 - attention
        copies = [features.clone(), attention.clone(), padded.clone(), stacked.clone()]
        args = (padded[..., ::2], stacked[:, ::2], ratio, eps)
        assert not args[0].is_contiguous() and not args[1].is_contiguous()
        for first, second in zip(_verify(solve, args), dense):
            if first.dtype == torch.int64:
                assert torch.equal(first, second)
            else:
                _close(first, second)
        for original, copy in zip((features, attention, padded, stacked), copies):
            assert torch.equal(original, copy), "input mutated"


def check_seeded_oracle_sweep(solve):
    # (batch, tokens, channels, ratio, eps, dtype)
    table = [
        (1, 4, 2, 0.5, 1e-12, torch.float64),
        (2, 9, 4, 1 / 3, 1e-12, torch.float32),
        (3, 8, 3, 0.25, 1e-9, torch.float64),
        (1, 12, 5, 0.6, 1e-12, torch.float32),
        (2, 6, 1, 0.5, 1e-12, torch.float64),
        (3, 7, 6, 3 / 7, 1e-6, torch.float32),
        (2, 5, 4, 1.0, 1e-12, torch.float64),
        (1, 16, 3, 0.125, 1e-12, torch.float32),
        (2, 11, 2, 0.5, 1e-12, torch.float64),
        (3, 13, 4, 0.75, 1e-30, torch.float32),
    ]
    for index, (b, n, c, ratio, eps, dtype) in enumerate(table):
        _verify(solve, _case(b=b, n=n, c=c, ratio=ratio, eps=eps, dtype=dtype,
                             seed=101 + 7 * index, coarse=index % 2 == 0))


def check_invalid_inputs(solve):
    features, attention, ratio, eps = _example()
    broken = [
        (features, attention, 0.0, 1e-12),
        (features, attention, -0.5, 1e-12),
        (features, attention, 1.5, 1e-12),
        (features, attention, float("nan"), 1e-12),
        (features, attention, "0.5", 1e-12),
        (features, attention, True, 1e-12),
        (features, attention, torch.tensor(0.5), 1e-12),
        (features, attention, 0.1, 1e-12),
        (features, attention, 1 / 6, 1e-12),
        (features, attention, 0.5, 0.0),
        (features, attention, 0.5, -1e-12),
        (features, attention, 0.5, float("nan")),
        (features, attention, 0.5, "1e-12"),
        (features, attention, 0.5, True),
        (features, attention[:, :4], 0.5, 1e-12),
        (features, attention.squeeze(0), 0.5, 1e-12),
        (features[0], attention, 0.5, 1e-12),
        (features, attention.repeat(2, 1), 0.5, 1e-12),
        (torch.empty((0, 3, 2), dtype=torch.float64), torch.empty((0, 3), dtype=torch.float64),
         0.5, 1e-12),
        (torch.empty((1, 0, 2), dtype=torch.float64), torch.empty((1, 0), dtype=torch.float64),
         0.5, 1e-12),
        (torch.empty((1, 3, 0), dtype=torch.float64), torch.full((1, 3), 0.5, dtype=torch.float64),
         0.5, 1e-12),
        (features.to(torch.float16), attention.to(torch.float16), 0.5, 1e-12),
        (features.to(torch.int32), attention, 0.5, 1e-12),
        (features, attention.to(torch.float32), 0.5, 1e-12),
        (features * float("inf"), attention, 0.5, 1e-12),
        (features, attention + float("nan"), 0.5, 1e-12),
        ("features", attention, 0.5, 1e-12),
    ]
    for args in broken:
        try:
            solve(*args)
        except ValueError:
            continue
        except AssertionError as exc:
            raise AssertionError(f"expected ValueError, got AssertionError: {exc}")
        raise AssertionError(f"expected ValueError for {args[2:]}")
    # a single patch cannot survive any ratio below 1
    tiny = (torch.randn(1, 1, 2, dtype=torch.float64), torch.full((1, 1), 0.5, dtype=torch.float64))
    for bad in (0.5, 0.9, 0.999):
        try:
            solve(tiny[0], tiny[1], bad)
        except ValueError:
            pass
        else:
            raise AssertionError("int(1 * 0.5) == 0 must raise")
    solve(tiny[0], tiny[1], 1.0)


check_example_two_kept_patches.example = True
check_example_zero_norm_single_keep.example = True
