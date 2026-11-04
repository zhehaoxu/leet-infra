"""GRESO (arXiv:2506.02177, official Infini-AI-Lab/GRESO) -- pre-rollout
selective filtering ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="id": data ids from (data_sources, indices) pairs
- mode="trace": replay profiling ops and return a reward trace
- mode="skip": single-example skip decision (kwargs: ops, current_epoch,
  data_id, variant in {easy_example, easy_backoff, hard_backoff,
  easy_probabilistic, hard_probabilistic}, plus variant params; random
  draws are supplied via ``rand_values`` / ``backoff``)
- mode="filter": whole-batch all_probabilistic filtering (kwargs: ops,
  current_epoch, batch, p_easy, p_hard, rand_values) -> (selected, log,
  counts)
- mode="refill": adaptive sampling bs estimate (kwargs: reminder,
  zero_variance_ratio, n, default_bs[, full_sampling_bs, min_bs])
- mode="adapt": p_easy / p_hard exploration adaptation (kwargs: epoch,
  hard_ratio, easy_ratio, p_hard, p_easy[, target_zero_variance, min_p,
  max_p])
- mode="zero_variance": zero-advantage group flags + kept flat indices
  (kwargs: rewards, n)

Random sources are fully parameterized (official torch.randint /
torch.rand) and consumed lazily -- a check only draws when it reaches the
random comparison (the hard "solvable" gate short-circuits before any
draw).
"""
