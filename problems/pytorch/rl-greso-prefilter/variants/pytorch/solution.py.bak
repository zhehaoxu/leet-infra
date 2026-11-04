"""GRESO (arXiv:2506.02177, official Infini-AI-Lab/GRESO) -- pre-rollout
selective filtering ledger, transcribed from
verl/trainer/ttis/presampling_selector.py::DataProfiler,
verl/trainer/ttis/ttis.py::filter_zero_advantage and the adaptive-sampling /
p-adaptation tail of verl/trainer/ppo/ray_trainer.py.

The judge injects a single ``solve``; all entry points are dispatched by the
``mode`` keyword. Random sources (official torch.randint / torch.rand) are
fully parameterized as values supplied by the caller, so the whole ledger is
deterministic.

Official quirks kept verbatim:
- "easy" means a run of consecutive trailing rewards >= 0.98; "hard" means
  the latest reward <= 0.11 (a solvable example, latest > 0.11, is never
  hard-skipped) followed by a run of consecutive trailing rewards <= 0.11;
- the backoff window is randint(0, k * count) anchored at the trace's max
  epoch (all entries, not just the run);
- the probabilistic skip probability is 1 - max(p ** count, 0.01) with no
  count == 0 early return (count 0 gives probability 0);
- easy_example_skip matches rewards exactly equal to 1 within an epoch
  window n (-1 means forever, internally 10e7);
- data ids are f"{data_source}_{index}";
- the refill estimate is int(remaining / (1 - zero_variance_ratio) / n
  * 1.25), clamped to [64, sampling_batch_size], with last_sampling set
  when the estimate falls to the default;
- p_easy / p_hard adapt by +-0.01 toward targets target/3 and target*2/3
  (hard keeps more exploration headroom), clamped to [0.05, 0.95], and are
  frozen at epoch 0.
"""


def make_profiler():
    return {"data": {}}


def add_reward(prof, epoch, data_id, reward):
    prof["data"].setdefault(data_id, []).append((epoch, reward))


def get_data_id_list(data_sources, indices):
    return ["%s_%s" % (s, i) for s, i in zip(data_sources, indices)]


def get_reward_trace(prof, data_id):
    return prof["data"].get(data_id, [])


def easy_example_skip(prof, current_epoch, data_id, n=-1):
    if data_id not in prof["data"]:
        return False
    reward_trace = prof["data"][data_id]
    if len(reward_trace) == 0:
        return False
    if n == -1:
        n = 10e7
    for i in range(0, len(reward_trace)):
        if reward_trace[i][0] + n >= current_epoch and \
                reward_trace[i][1] == 1:
            return True
    return False


def easy_linear_backoff_skip(prof, current_epoch, data_id, k, backoff):
    if data_id not in prof["data"]:
        return False
    reward_trace = prof["data"][data_id]
    if len(reward_trace) == 0:
        return False
    count = 0
    for i in range(len(reward_trace) - 1, -1, -1):
        if reward_trace[i][1] < 0.98:
            break
        count += 1
    if count == 0:
        return False
    max_epoch = max([reward[0] for reward in reward_trace])
    if max_epoch + backoff >= current_epoch:
        return True
    return False


def hard_linear_backoff_skip(prof, current_epoch, data_id, k, backoff):
    if data_id not in prof["data"]:
        return False
    reward_trace = prof["data"][data_id]
    if len(reward_trace) == 0:
        return False
    if reward_trace[-1][1] > 0.11:
        return False
    count = 0
    for i in range(len(reward_trace) - 1, -1, -1):
        if reward_trace[i][1] > 0.11:
            break
        count += 1
    if count == 0:
        return False
    max_epoch = max([reward[0] for reward in reward_trace])
    if max_epoch + backoff >= current_epoch:
        return True
    return False


def easy_probabilistic_skip(prof, current_epoch, data_id, p, rand_fn):
    if data_id not in prof["data"]:
        return False
    reward_trace = prof["data"][data_id]
    if len(reward_trace) == 0:
        return False
    count = 0
    for i in range(len(reward_trace) - 1, -1, -1):
        if reward_trace[i][1] < 0.98:
            break
        count += 1
    skip_probability = 1 - max(p ** count, 0.01)
    # official: torch.rand(1) is sampled lazily -- only when reached
    return rand_fn() < skip_probability


def hard_probabilistic_skip(prof, current_epoch, data_id, p, rand_fn):
    if data_id not in prof["data"]:
        return False
    reward_trace = prof["data"][data_id]
    if len(reward_trace) == 0:
        return False
    if reward_trace[-1][1] > 0.11:
        return False
    count = 0
    for i in range(len(reward_trace) - 1, -1, -1):
        if reward_trace[i][1] > 0.11:
            break
        count += 1
    skip_probability = 1 - max(p ** count, 0.01)
    return rand_fn() < skip_probability


def filter_all_probabilistic(prof, current_epoch, batch, p_easy, p_hard,
                             rand_fn):
    """Official filter_examples_all_probabilistic (batch = list of dicts
    with data_source / index; rand_fn consumed lazily, only by checks that
    actually reach the random draw)."""
    data_index = [row["index"] for row in batch]
    data_source = [row["data_source"] for row in batch]
    data_id_list = get_data_id_list(data_source, data_index)
    easy_examples_count = 0
    hard_examples_count = 0
    skip_log = []
    selected_index = []
    for index, data_id in enumerate(data_id_list):
        if easy_probabilistic_skip(prof, current_epoch, data_id, p_easy,
                                   rand_fn):
            easy_examples_count += 1
            skip_log.append([data_id, 0])
            continue
        if hard_probabilistic_skip(prof, current_epoch, data_id, p_hard,
                                   rand_fn):
            hard_examples_count += 1
            skip_log.append([data_id, 1])
            continue
        selected_index.append(index)
        skip_log.append([data_id, 2])
    return selected_index, skip_log, [easy_examples_count,
                                      hard_examples_count]


def adaptive_sampling_bs(reminder, zero_variance_ratio, n, default_bs,
                         full_sampling_bs=False, min_bs=64):
    """Official refill estimate; returns (bs, last_sampling, used_full)."""
    if full_sampling_bs:
        return default_bs, False, True
    estimated = int(reminder / (1 - zero_variance_ratio) / n * 1.25)
    last = estimated <= default_bs
    bs = min(estimated, default_bs)
    bs = max(bs, min_bs)
    return bs, last, False


def adapt_p(epoch, hard_ratio, easy_ratio, p_hard, p_easy,
            target_zero_variance=0.25, min_p=0.05, max_p=0.95):
    """Official exploration-rate adaptation; returns (p_hard, p_easy)."""
    if epoch <= 0:
        return p_hard, p_easy
    targeted_easy = target_zero_variance / 3
    targeted_hard = target_zero_variance * 2 / 3
    if hard_ratio >= targeted_hard:
        p_hard = max(p_hard - 0.01, min_p)
    else:
        p_hard = min(p_hard + 0.01, max_p)
    if easy_ratio >= targeted_easy:
        p_easy = max(p_easy - 0.01, min_p)
    else:
        p_easy = min(p_easy + 0.01, max_p)
    return p_hard, p_easy


def filter_zero_advantage(rewards, n):
    """Official filter_zero_advantage: a group whose rewards have
    max == min carries zero advantage; returns (flags, kept flat indices)."""
    groups = [rewards[i:i + n] for i in range(0, len(rewards), n)]
    zero_flag = [max(g) == min(g) for g in groups]
    keep = []
    for gi, zf in enumerate(zero_flag):
        if not zf:
            keep.extend([gi * n + j for j in range(n)])
    return zero_flag, keep


def _replay(prof, ops):
    """Replay [op, epoch, data_id(s), reward(s), ...] profiling ops."""
    for op in ops:
        kind = op[0]
        epoch = op[1]
        if kind == "add":
            data_ids = op[2]
            rewards = op[3]
            if not isinstance(data_ids, list):
                data_ids = [data_ids]
                rewards = [rewards]
            for did, r in zip(data_ids, rewards):
                add_reward(prof, epoch, did, r)


def solve(mode=None, **kw):
    if mode == "id":
        return get_data_id_list(kw["data_sources"], kw["indices"])
    if mode == "trace":
        prof = make_profiler()
        _replay(prof, kw["ops"])
        return get_reward_trace(prof, kw["data_id"])
    if mode == "skip":
        prof = make_profiler()
        _replay(prof, kw["ops"])
        variant = kw["variant"]
        ce = kw["current_epoch"]
        data_id = kw["data_id"]
        if variant == "easy_example":
            return easy_example_skip(prof, ce, data_id, kw.get("n", -1))
        if variant == "easy_backoff":
            return easy_linear_backoff_skip(prof, ce, data_id, kw["k"],
                                            kw["backoff"])
        if variant == "hard_backoff":
            return hard_linear_backoff_skip(prof, ce, data_id, kw["k"],
                                            kw["backoff"])
        if variant == "easy_probabilistic":
            rands = iter(kw["rand_values"])
            return easy_probabilistic_skip(prof, ce, data_id, kw["p"],
                                           lambda: next(rands))
        if variant == "hard_probabilistic":
            rands = iter(kw["rand_values"])
            return hard_probabilistic_skip(prof, ce, data_id, kw["p"],
                                           lambda: next(rands))
        raise ValueError("unknown skip variant: %s" % variant)
    if mode == "filter":
        prof = make_profiler()
        _replay(prof, kw["ops"])
        rands = iter(kw["rand_values"])
        return filter_all_probabilistic(prof, kw["current_epoch"],
                                        kw["batch"], kw["p_easy"],
                                        kw["p_hard"],
                                        lambda: next(rands))
    if mode == "refill":
        return adaptive_sampling_bs(kw["reminder"], kw["zero_variance_ratio"],
                                    kw["n"], kw["default_bs"],
                                    kw.get("full_sampling_bs", False),
                                    kw.get("min_bs", 64))
    if mode == "adapt":
        return adapt_p(kw["epoch"], kw["hard_ratio"], kw["easy_ratio"],
                       kw["p_hard"], kw["p_easy"],
                       kw.get("target_zero_variance", 0.25),
                       kw.get("min_p", 0.05), kw.get("max_p", 0.95))
    if mode == "zero_variance":
        return filter_zero_advantage(kw["rewards"], kw["n"])
    raise ValueError("unknown mode: %s" % mode)
