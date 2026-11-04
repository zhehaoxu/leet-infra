"""KTAE (arXiv:2505.16826, official ZNLP/KTAE) -- token-level advantage
ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="distribution": token id -> number of distinct rows containing it
  (kwargs: tensor, mask)
- mode="appear_table": 2x2 contingency table per token id
  (kwargs: responses_ids, mask, rewards[, max_token_num])
- mode="fisher": formatted Fisher exact signal exp(-2p), p==1 -> 0
  (kwargs: table)
- mode="ig": information gain of correct/wrong given token presence
  (kwargs: table)
- mode="tf": BM25-saturated term frequencies (kwargs: correct,
  correct_mask, wrong, wrong_mask, table)
- mode="cohen": arcsine-sqrt distance of the two in-group rates
  (kwargs: table)
- mode="key_tokens": full key-token weight table (kwargs: responses_ids,
  mask, rewards[, alpha, beta_ig, gamma_tf, top, max_token_num])
- mode="advantage": grouped standardized outcome + per-token weights
  fused over the eos mask (kwargs: token_level_rewards, responses,
  eos_mask, index[, epsilon])

Tensors are float32 torch tensors. Mind the official quirks: the key-token
table is sized max_token_num + 1 with id 151643 unconditionally zeroed,
single-row groups allocate their zero table as responses.max() (no +1),
and torch.std is unbiased.
"""
