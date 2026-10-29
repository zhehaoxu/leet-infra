def solve(message_sizes, latencies_us, bandwidths_gbps):
    """Select calibrated communication candidates using the described model.

    Inputs (ordinary Python lists; valid inputs are guaranteed):
        message_sizes: list[int], 0 <= N <= 1024; 0 <= S <= 10**15 bytes.
        latencies_us: list[list[float]], shape (2, 3); values in [0.01, 10**6],
            in microseconds.
        bandwidths_gbps: list[list[float]], shape (2, 3); values in [0.01, 10**4],
            in decimal GB/s (gigabytes, not gigabits).
        Table rows: Ring, Tree. Columns: LL, LL128, Simple.
        The supplied values already include topology/world-size effects.

    Return a dict with exactly these keys, using Python lists, not tensors:
        times_us: list[list[float]], shape (N, 6), positive model times.
        best: list[int], length N, indices in [0, 5]; exact ties use the
            smallest index in row-major candidate order.
        thresholds_bytes: list[list[float | None]], shape (6, 6).
            Nonnegative real byte crossings, not rounded to integers;
            diagonal, parallel/identical pairs, and negative crossings
            are None. The matrix is symmetric.
        achieved_gbps: list[float], length N, nonnegative decimal GB/s.
        utilization: list[float], length N, in [0, 1] allowing rounding.
            Both throughput metrics are zero for S = 0.
        For N = 0, per-message lists are empty but thresholds are computed.

    Do not mutate inputs. Standard Python floating-point arithmetic suffices;
    no tensor library, hardware access, or input-validation code is required.
    """
    raise NotImplementedError
