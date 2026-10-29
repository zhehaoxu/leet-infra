def solve(n_stages, n_microbatches, n_virtual, n_layers, act_bytes):
    """Return a deterministic V-placement training timeline and activation ledger."""
    # Inputs: p=n_stages in [1,8], m=n_microbatches in [1,32], v=n_virtual in [1,4].
    # n_layers in [p*v,8192], divisible by p*v; act_bytes is a finite float >= 0.
    # Raise ValueError if n_layers is not divisible by p*v; other inputs are valid.
    # Return dict: timeline (p x T list, entries None or ('F'/'B', chunk, microbatch)),
    # busy_table (p x T bool lists), makespan (int T), bubble/util (float),
    # peak_activation_bytes/flush_peak_activation_bytes (p float lists),
    # memory_saved_bytes (float, sum of per-stage flush peaks minus scheduled peaks).
    raise NotImplementedError
