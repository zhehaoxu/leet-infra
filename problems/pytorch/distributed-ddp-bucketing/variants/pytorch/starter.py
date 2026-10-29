def solve(param_sizes, bucket_cap_mb, world_size, ready_times):
    """Model static buckets and globally synchronized eligibility, not a network.

    Inputs (guaranteed valid; no validation of out-of-contract inputs required):
        param_sizes: list of N positive integer gradient sizes in bytes, in
                     parameter-index order. An empty list is allowed.
        bucket_cap_mb: finite float in MiB despite the argument name;
                       cap = int(bucket_cap_mb * 1024**2) is positive.
                       Thus fractional bytes are rounded down, not up.
        world_size: positive integer W, the number of participating ranks.
        ready_times: list of W lists, each of length N, of nonnegative finite
                     floats. Entry [r][i] is parameter i's gradient-ready time
                     on rank r, in a shared arbitrary time unit.

    Packing is independent of readiness: visit indices N-1, ..., 0. Append a
    gradient if the current bucket still fits (equality fits); otherwise seal
    that bucket and start the next. A gradient larger than cap is a singleton,
    sealing any preceding partial bucket. Never split or reorder a gradient.

    Return a dict with exactly these keys, using Python lists:
        buckets: K lists of parameter indices in static visitation order.
        bucket_ready_times: W x K; [r][k] is the maximum ready_times[r][i]
                            among members i of bucket k.
        global_ready_times: K values; [k] is the maximum bucket readiness over
                            all W ranks. Every rank participates in every bucket.
        enqueue_times: K values; [k] is max(global_ready_times[:k+1]), so a later
                       bucket cannot overtake an earlier one. Equal times are
                       allowed. These are global synchronized eligibility
                       times, NOT actual local enqueue or network completion.

    Empty param_sizes yields buckets=[], bucket_ready_times=[[] for each rank],
    global_ready_times=[], enqueue_times=[]. Do not mutate inputs. There are no
    communication durations, real collectives, or root/start-owner ranks.
    """
    raise NotImplementedError
