"""KTAE (arXiv:2505.16826, official ZNLP/KTAE) -- token-level advantage
ledger, transcribed from verl/workers/actor/compute_key_tokens.py and
verl/trainer/ppo/core_algos.py::compute_ktae_outcome_advantage_and_keytokens.

The judge injects a single ``solve``; all entry points are dispatched by the
``mode`` keyword. Tensors are float32 torch tensors (official dtype).

Official quirks kept verbatim:
- the key-token table is sized ``max_token_num + 1`` where the official
  caller passes ``responses.max().item()`` (max token id + 1), and id
  ``151643`` is unconditionally zeroed -- with a small vocab that line
  raises IndexError (native behaviour, pinned by a test case);
- ``torch.std`` is unbiased (n-1 denominator);
- the advantage fuses the standardized outcome score broadcast over the
  eos mask with a per-token weight table indexed by token id;
- ``index`` must hash by value (list/numpy) -- the official trainer passes
  a numpy array.
"""
import math

import torch


def _clamp01(x):
    return torch.clamp(x, min=0, max=1)


def _compute_cohen(a, b):
    """Official compute_Cohen: arcsine-sqrt distance of two rates."""
    a = _clamp01(a)
    b = _clamp01(b)
    h = torch.asin(torch.sqrt(a)) - torch.asin(torch.sqrt(b))
    return h


def _log_comb(n, k):
    return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)


def _fisher_exact_test(input_tensor):
    """Official fisher_exact_test: numeric Fisher exact p-values via lgamma."""
    a, b, c, d = torch.unbind(input_tensor, dim=1)
    R1 = a + b
    R2 = c + d
    C1 = a + c
    n = R1 + R2
    k_min = torch.max(torch.tensor(0, dtype=torch.float32), R1 + C1 - n)
    k_max = torch.min(R1, C1)
    max_k = torch.max(k_max)
    k_values = torch.arange(max_k + 1, dtype=torch.float32,
                            device=input_tensor.device)
    k_grid = torch.tile(k_values, (input_tensor.size(0), 1))
    mask = (k_grid >= k_min.unsqueeze(1)) & (k_grid <= k_max.unsqueeze(1))
    C1_grid = C1.unsqueeze(1).expand_as(k_grid)
    R1_grid = R1.unsqueeze(1).expand_as(k_grid)
    n_grid = n.unsqueeze(1).expand_as(k_grid)
    k = k_grid
    log_p_k = (_log_comb(C1_grid, k)
               + _log_comb(n_grid - C1_grid, R1_grid - k)
               - _log_comb(n_grid, R1_grid))
    p_k = torch.exp(log_p_k) * mask.float()
    log_p_obs = (_log_comb(C1, a) + _log_comb(n - C1, R1 - a)
                 - _log_comb(n, R1))
    p_obs = torch.exp(log_p_obs)
    p_values = torch.sum(torch.where((p_k <= p_obs.unsqueeze(1)) & mask,
                                     p_k, torch.tensor(0.)), dim=1)
    return p_values


def _calculate_ig(table_tensor):
    """Official calculate_ig: information gain of correct/wrong given
    token presence, in log2."""
    t_a = table_tensor[:, 0]
    t_b = table_tensor[:, 1]
    t_c = table_tensor[:, 2]
    t_d = table_tensor[:, 3]
    t_n = t_a + t_b + t_c + t_d

    def safe_log(x):
        return torch.where(x > 0, torch.log2(x),
                           torch.tensor(0.0, dtype=torch.float32,
                                        device=x.device))

    def entropy(p):
        return -p * safe_log(p)

    p_correct = (t_a + t_c) / t_n
    p_error = (t_b + t_d) / t_n
    h_y = entropy(p_correct) + entropy(p_error)

    def conditional_entropy(x1, x2):
        total = x1 + x2
        return torch.where(total > 0,
                           entropy(x1 / total) + entropy(x2 / total),
                           torch.tensor(0.0, dtype=torch.float32,
                                        device=x1.device))

    h_y_x = ((t_a + t_b) / t_n) * conditional_entropy(t_a, t_b) + \
            ((t_c + t_d) / t_n) * conditional_entropy(t_c, t_d)
    ig = h_y - h_y_x
    ig[t_n == 0] = 0
    return ig


def _compute_tf(correct_tensor, correct_mask, wrong_tensor, wrong_mask,
                appear_table_tensor):
    """Official compute_tf: BM25-style saturated term frequencies of each
    token inside the correct / wrong pools."""
    k1 = 2.0
    b = 0.5
    correct_tensor_unmasked = correct_tensor[correct_mask == 1]
    wrong_tensor_unmasked = wrong_tensor[wrong_mask == 1]
    total_correct = correct_tensor_unmasked.numel()
    total_wrong = wrong_tensor_unmasked.numel()
    avg_len = (total_correct + total_wrong) / 2
    corrcet_k = k1 * (1 - b + b * (total_correct / avg_len))
    wrong_k = k1 * (1 - b + b * (total_wrong / avg_len))
    correct_hist = torch.bincount(correct_tensor_unmasked,
                                  minlength=appear_table_tensor.size(0))
    wrong_hist = torch.bincount(wrong_tensor_unmasked,
                                minlength=appear_table_tensor.size(0))
    tf_tensor = torch.stack(
        [(correct_hist.float() / total_correct) * (k1 + 1) /
         ((correct_hist.float() / total_correct) + corrcet_k),
         (wrong_hist.float() / total_wrong) * (k1 + 1) /
         ((wrong_hist.float() / total_wrong) + wrong_k)],
        dim=1).to(dtype=torch.float32, device=appear_table_tensor.device)
    return tf_tensor


def _get_distribution(tensor, mask):
    """Official get_distribution: token id -> number of distinct rows of
    ``tensor`` in which the token appears (mask == 1 cells only)."""
    appear_distribution = {}
    m, n = tensor.shape
    row_indices = torch.arange(m)[:, None].expand(m, n)
    flat_values = tensor.flatten()
    flat_rows = row_indices.flatten()
    flat_mask = mask.flatten()
    masked_values = flat_values[flat_mask == 1]
    masked_rows = flat_rows[flat_mask == 1]
    unique_values, inverse_indices = torch.unique(masked_values,
                                                  return_inverse=True)
    for i, val in enumerate(unique_values.tolist()):
        current_rows = masked_rows[inverse_indices == i]
        unique_rows = torch.unique(current_rows)
        appear_distribution[val] = len(unique_rows)
    return appear_distribution


def _build_appear_table(correct_tensor, wrong_tensor, correct_mask,
                        wrong_mask, max_token_num):
    """Official appear-table construction inside compute_key_tokens."""
    correct_dist = _get_distribution(correct_tensor, correct_mask)
    wrong_dist = _get_distribution(wrong_tensor, wrong_mask)
    appear_table_tensor = torch.zeros((max_token_num + 1, 4),
                                      dtype=torch.float32,
                                      device=correct_tensor.device)
    tokenid_set = set(torch.cat([correct_tensor.flatten(),
                                 wrong_tensor.flatten()]).tolist())
    total_correct = correct_tensor.size(0)
    total_wrong = wrong_tensor.size(0)
    for tokenid in tokenid_set:
        correct_count = correct_dist.get(tokenid, 0)
        wrong_count = wrong_dist.get(tokenid, 0)
        if correct_count + wrong_count != 0:
            appear_table_tensor[tokenid] = torch.tensor(
                [correct_count, wrong_count,
                 total_correct - correct_count,
                 total_wrong - wrong_count],
                dtype=torch.float32, device=appear_table_tensor.device)
    return appear_table_tensor


def _format_fisher(fisher_pvalue):
    """Official formatting: exp(-2p), with p == 1 squashed to 0."""
    return torch.where(
        fisher_pvalue != 1.0, torch.exp(-2 * fisher_pvalue),
        torch.tensor(0.0, dtype=torch.float32, device=fisher_pvalue.device))


def _key_token_result(appear_table_tensor, correct_tensor, correct_mask,
                      wrong_tensor, wrong_mask, alpha, beta_ig, gamma_tf,
                      top, max_token_num):
    """Official compute_key_tokens tail: signal fusion -> sigmoid rescale
    -> hardcoded zero at id 151643."""
    if correct_tensor.size(0) == 0 or wrong_tensor.size(0) == 0:
        return torch.zeros(max_token_num + 1, dtype=torch.float32)
    ig_val = (_calculate_ig(appear_table_tensor) if beta_ig else torch.zeros(
        max_token_num + 1, dtype=torch.float32,
        device=appear_table_tensor.device))
    if alpha:
        fisher_pvalue = _fisher_exact_test(appear_table_tensor)
        format_fisher_pvalue = _format_fisher(fisher_pvalue)
    else:
        format_fisher_pvalue = torch.zeros(
            max_token_num + 1, dtype=torch.float32,
            device=appear_table_tensor.device)
    if gamma_tf:
        tf_tensor = _compute_tf(correct_tensor, correct_mask, wrong_tensor,
                                wrong_mask, appear_table_tensor)
        x = _compute_cohen(
            appear_table_tensor[:, 0] /
            (appear_table_tensor[:, 0] + appear_table_tensor[:, 2] + 1e-10),
            appear_table_tensor[:, 1] /
            (appear_table_tensor[:, 1] + appear_table_tensor[:, 3] + 1e-10))
        x += (tf_tensor[:, 0] / (tf_tensor[:, 1] + 1e-5) -
              tf_tensor[:, 1] / (tf_tensor[:, 0] + 1e-5))
    else:
        x = _compute_cohen(
            appear_table_tensor[:, 0] /
            (appear_table_tensor[:, 0] + appear_table_tensor[:, 2] + 1e-10),
            appear_table_tensor[:, 1] /
            (appear_table_tensor[:, 1] + appear_table_tensor[:, 3] + 1e-10))
    result = ((alpha * format_fisher_pvalue + beta_ig * ig_val) * x) / \
             (alpha + beta_ig)
    result = (torch.sigmoid(result) - 0.5) * 2 * top
    result[151643] = 0.
    return result


def solve(mode=None, **kw):
    if mode == "distribution":
        return _get_distribution(kw["tensor"], kw["mask"])
    if mode == "appear_table":
        responses = kw["responses_ids"]
        mask = kw["mask"]
        rewards = kw["rewards"]
        max_token_num = kw.get("max_token_num")
        if max_token_num is None:
            max_token_num = int(responses.max())
        return _build_appear_table(
            responses[rewards > 0.0], responses[rewards <= 0.0],
            mask[rewards > 0.0], mask[rewards <= 0.0], max_token_num)
    if mode == "fisher":
        return _format_fisher(_fisher_exact_test(kw["table"]))
    if mode == "ig":
        return _calculate_ig(kw["table"])
    if mode == "cohen":
        t = kw["table"]
        return _compute_cohen(
            t[:, 0] / (t[:, 0] + t[:, 2] + 1e-10),
            t[:, 1] / (t[:, 1] + t[:, 3] + 1e-10))
    if mode == "tf":
        return _compute_tf(kw["correct"], kw["correct_mask"], kw["wrong"],
                           kw["wrong_mask"], kw["table"])
    if mode == "key_tokens":
        responses = kw["responses_ids"]
        mask = kw["mask"]
        rewards = kw["rewards"]
        max_token_num = kw.get("max_token_num")
        if max_token_num is None:
            max_token_num = int(responses.max())
        correct_outputs = responses[rewards > 0.0]
        correct_outputs_mask = mask[rewards > 0.0]
        wrong_outputs = responses[rewards <= 0.0]
        wrong_outputs_mask = mask[rewards <= 0.0]
        table = _build_appear_table(correct_outputs, wrong_outputs,
                                    correct_outputs_mask,
                                    wrong_outputs_mask, max_token_num)
        return _key_token_result(
            table, correct_outputs, correct_outputs_mask, wrong_outputs,
            wrong_outputs_mask, kw.get("alpha", 1.0), kw.get("beta_ig", 1.0),
            kw.get("gamma_tf", 1.0), kw.get("top", 1.0), max_token_num)
    if mode == "advantage":
        token_level_rewards = kw["token_level_rewards"]
        responses = kw["responses"]
        eos_mask = kw["eos_mask"]
        index = kw["index"]
        epsilon = kw.get("epsilon", 1e-6)
        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)
        id2score = {}
        id2reponses = {}
        id2mask = {}
        id2mean = {}
        id2std = {}
        id2key_token = {}
        bsz = token_level_rewards.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2score.setdefault(idx, []).append(scores[i])
            if idx in id2reponses:
                id2reponses[idx] = torch.cat(
                    (id2reponses[idx], responses[i].unsqueeze(0)), dim=0)
                id2mask[idx] = torch.cat(
                    (id2mask[idx], eos_mask[i].unsqueeze(0)), dim=0)
            else:
                id2reponses[idx] = responses[i].unsqueeze(0)
                id2mask[idx] = eos_mask[i].unsqueeze(0)
        for idx in id2score:
            reponses_per_q = id2reponses[idx]
            mask_per_q = id2mask[idx]
            score_per_q = id2score[idx]
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
                id2key_token[idx] = torch.zeros(
                    [responses.max().item()], device=responses.device)
            elif len(id2score[idx]) > 1:
                format_score_per_q = torch.tensor(score_per_q)
                id2mean[idx] = torch.mean(format_score_per_q)
                id2std[idx] = torch.std(format_score_per_q)
                max_token_num = int(responses.max().item())
                table = _build_appear_table(
                    reponses_per_q[format_score_per_q > 0.0],
                    reponses_per_q[format_score_per_q <= 0.0],
                    mask_per_q[format_score_per_q > 0.0],
                    mask_per_q[format_score_per_q <= 0.0], max_token_num)
                id2key_token[idx] = _key_token_result(
                    table, reponses_per_q[format_score_per_q > 0.0],
                    mask_per_q[format_score_per_q > 0.0],
                    reponses_per_q[format_score_per_q <= 0.0],
                    mask_per_q[format_score_per_q <= 0.0],
                    1.0, 1.0, 1.0, 1.0, max_token_num)
            else:
                raise ValueError("no score in prompt index: %s" % idx)
        means = torch.tensor([float(id2mean[index[i]]) for i in range(bsz)],
                             device=scores.device)
        stds = torch.tensor([float(id2std[index[i]]) for i in range(bsz)],
                            device=scores.device)
        scores = (scores - means) / (stds + epsilon)
        format_weights = [id2key_token[index[i]][responses[i]].unsqueeze(0)
                          for i in range(bsz)]
        all_weight = torch.cat(format_weights, dim=0)
        scores = (scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
                  + all_weight * eos_mask)
        return scores
    raise ValueError("unknown mode: %s" % mode)
