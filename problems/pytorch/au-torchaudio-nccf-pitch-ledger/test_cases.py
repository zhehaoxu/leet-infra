#!/usr/bin/env python3
"""858 judge cases: torchaudio NCCF pitch-detection ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md).  No torch, no import
from the solution module, no global RNG.
"""

WF = [1., 0., -1., 0.] * 10          # 40 samples
NCCF0 = [0.0, -0.24999999949999996, 0.0, 0.24999999949999996,
         0.0, -0.24999999949999996, 0.0, 0.24999999949999996]
NCCF4 = [0.0, -0.24999999946132487, 0.0, 0.24999999939644652,
         0.0, -0.24999999924999994, 0.0, 0.0]


def _near(got, exp, tol=1e-8):
    if isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), (got, exp)
        for a, b in zip(got, exp):
            _near(a, b, tol)
    elif isinstance(exp, float):
        assert abs(got - exp) <= tol, (got, exp)
    else:
        assert got == exp, (got, exp)


def _err(fn, key):
    try:
        fn()
    except ValueError as e:
        assert key in str(e), "expected %r in %r" % (key, str(e))
        return
    raise AssertionError("expected ValueError containing %r" % key)


# ---------------------------------------------------------------------------
def check_nccf(solve):
    """Framing meta and the fourth-power-normalized NCCF values."""
    r = solve("nccf", waveform=WF, sample_rate=800,
              frame_time=0.01, freq_low=100)
    _near(r["shape"], [5, 8])
    assert r["lags"] == 8 and r["frame_size"] == 8
    assert r["num_frames"] == 5 and r["pad"] == 8
    assert r["padded_length"] == 48 and r["waveform_length"] == 40
    _near(r["nccf"][0], NCCF0)
    _near(r["nccf"][1], NCCF0)
    _near(r["nccf"][4], NCCF4)
    # different parameters change lags / frame_size / frames / pad
    r2 = solve("nccf", waveform=[0.5] * 30, sample_rate=1000,
               frame_time=0.005, freq_low=250)
    assert r2["lags"] == 4 and r2["frame_size"] == 5
    assert r2["num_frames"] == 6 and r2["pad"] == 4
    _near(r2["shape"], [6, 4])
    # batched leading dimension stays independent
    r3 = solve("nccf", waveform=[WF, [v * 2 for v in WF]],
               sample_rate=800, frame_time=0.01, freq_low=100)
    _near(r3["shape"], [2, 5, 8])
    _near(r3["nccf"][0], r["nccf"])
    # The fourth-power denominator is NOT scale invariant: doubling the
    # waveform divides every NCCF value by four (plus an epsilon drift).
    _near(r3["nccf"][1], [[v / 4.0 for v in row] for row in r["nccf"]],
          tol=1e-9)
    _near(r3["nccf"][1][0][1], -0.062499999937499995, tol=1e-12)


def check_nccf_domain(solve):
    """NCCF input contracts."""
    _err(lambda: solve("nccf", waveform=[], sample_rate=800,
                       frame_time=0.01, freq_low=100),
         "waveform must be a non-empty numeric array")
    _err(lambda: solve("nccf", waveform=["a"], sample_rate=800,
                       frame_time=0.01, freq_low=100),
         "waveform must be a non-empty numeric array")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=True,
                       frame_time=0.01, freq_low=100),
         "sample_rate must be an integer")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=0,
                       frame_time=0.01, freq_low=100),
         "sample_rate must be at least 1")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=800,
                       frame_time=0, freq_low=100),
         "frame_time must be a positive number")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=800,
                       frame_time=0.01, freq_low=2.5),
         "freq_low must be an integer")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=800,
                       frame_time=0.01, freq_low=0),
         "freq_low must be at least 1")


def check_nccf_quirk(solve):
    """The source divides by two squared norms, not sqrt(E1*E2)."""
    r = solve("nccf", waveform=WF, sample_rate=800,
              frame_time=0.01, freq_low=100)
    _near(r["nccf"][0][3], 0.2499999995)
    assert abs(r["nccf"][0][3] - 1.0) > 0.5        # docstring would give 1.0
    assert r["nccf"][0][3] != 0.25                 # the 1e-9 epsilon shows
    for k in (0, 2, 4, 6):
        assert abs(r["nccf"][0][k]) < 1e-12
    # the tail frame sees the zero padding
    assert r["nccf"][4][7] == 0.0
    _near(r["nccf"][4][3], 0.24999999939644652)


# ---------------------------------------------------------------------------
def check_combine_max(solve):
    """Strict > threshold mask selects a's value AND index."""
    r = solve("combine_max", a_values=[0.99, 0.5, 1.0], a_indices=[2, 3, 4],
              b_values=[1.0, 0.9, 0.5], b_indices=[7, 8, 9])
    assert r["mask"] == [False, False, True]
    _near(r["values"], [1.0, 0.9, 1.0])
    assert r["indices"] == [7, 8, 4]
    _near(r["thresh"], 0.99)
    # exactly 0.99 * b is NOT selected
    r2 = solve("combine_max", a_values=[0.99], a_indices=[1],
               b_values=[1.0], b_indices=[2])
    assert r2["mask"] == [False] and r2["indices"] == [2]
    # custom threshold changes the boundary
    r3 = solve("combine_max", a_values=[0.8], a_indices=[1],
               b_values=[1.0], b_indices=[2], thresh=0.5)
    assert r3["mask"] == [True] and r3["indices"] == [1]
    _near(r3["thresh"], 0.5)


def check_combine_max_domain(solve):
    """combine_max input contracts."""
    _err(lambda: solve("combine_max", a_values=[1.], a_indices=[1],
                       b_values=[1., 2.], b_indices=[1, 2]),
         "a and b must have the same length")
    _err(lambda: solve("combine_max", a_values=[1.], a_indices=[1],
                       b_values=[1.], b_indices=[1], thresh=0),
         "thresh must be a positive number")
    _err(lambda: solve("combine_max", a_values=[], a_indices=[],
                       b_values=[], b_indices=[]),
         "a_values must be a non-empty numeric array")
    _err(lambda: solve("nope"), "unknown mode")


# ---------------------------------------------------------------------------
def check_find_max(solve):
    """Half-window preference and the two empirical offsets."""
    r = solve("find_max", nccf=[[0., 0.9, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]],
              sample_rate=800, freq_high=800)
    assert r["lag_min"] == 1 and r["half_size"] == 4
    assert r["indices"] == [5]                    # global best, 0 + 1 + 1 = 2? see note
    _near(r["best_values"], [1.0])
    assert r["best_indices"] == [3]
    _near(r["half_values"], [0.9])
    assert r["half_indices"] == [0]
    r2 = solve("find_max", nccf=[[0., 0.995, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1]],
               sample_rate=800, freq_high=800)
    assert r2["indices"] == [2]                   # half preferred: 0 + 1 + 1
    # double calibration: relative index 0, lag_min 2, then +1
    r3 = solve("find_max", nccf=[[0., 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]],
               sample_rate=800, freq_high=400)
    assert r3["lag_min"] == 2 and r3["indices"] == [3]
    # real NCCF rows
    rows = solve("nccf", waveform=WF, sample_rate=800,
                 frame_time=0.01, freq_low=100)["nccf"]
    r4 = solve("find_max", nccf=rows, sample_rate=800, freq_high=800)
    assert r4["indices"] == [4, 4, 4, 4, 4]
    assert r4["best_indices"] == [2, 2, 2, 2, 2]


def check_find_max_domain(solve):
    """find_max input contracts."""
    _err(lambda: solve("find_max", nccf=[0.1, 0.2, 0.3],
                       sample_rate=800, freq_high=800),
         "nccf must have at least 2 dimensions")
    _err(lambda: solve("find_max", nccf=[[0., 0.9, 0.1, 0.1]],
                       sample_rate=800, freq_high=400),
         "nccf last dimension is too short for freq_high")
    _err(lambda: solve("find_max", nccf=[[0.] * 8], sample_rate=True,
                       freq_high=800),
         "sample_rate must be an integer")
    _err(lambda: solve("find_max", nccf=[[0.] * 8], sample_rate=800,
                       freq_high=True),
         "freq_high must be an integer")
    _err(lambda: solve("find_max", nccf=[["a"] * 8], sample_rate=800,
                       freq_high=800),
         "nccf must be a non-empty numeric array")


# ---------------------------------------------------------------------------
def check_median_smooth(solve):
    """Left replicate padding, unfolding and lower median."""
    r = solve("median_smooth", indices=[1., 2., 3., 4., 5.], win_length=5)
    assert r["pad_length"] == 2 and r["shape"] == [3]
    _near(r["padded"], [1., 1., 1., 2., 3., 4., 5.])
    _near(r["roll"], [[1., 1., 1., 2., 3.],
                      [1., 1., 2., 3., 4.],
                      [1., 2., 3., 4., 5.]])
    _near(r["smoothed"], [1., 2., 3.])
    # odd window with a non-monotonic sequence
    r2 = solve("median_smooth", indices=[9., 1., 8., 2., 7., 3., 6.],
               win_length=3)
    assert r2["pad_length"] == 1 and r2["shape"] == [6]
    _near(r2["padded"], [9., 9., 1., 8., 2., 7., 3., 6.])
    _near(r2["smoothed"], [9., 8., 2., 7., 3., 6.])
    # batched leading dimension
    r3 = solve("median_smooth",
               indices=[[1., 2., 3., 4.], [4., 3., 2., 1.]],
               win_length=3)
    _near(r3["shape"], [2, 3])
    _near(r3["smoothed"][0], [1., 2., 3.])


def check_median_smooth_domain(solve):
    """median_smooth input contracts."""
    _err(lambda: solve("median_smooth", indices=[1., 2., 3., 4.],
                       win_length=2.5),
         "win_length must be an integer")
    _err(lambda: solve("median_smooth", indices=[1., 2., 3., 4.],
                       win_length=2),
         "win_length must be at least 3")
    _err(lambda: solve("median_smooth", indices=[1., 2.], win_length=5),
         "indices length is too short for win_length")
    _err(lambda: solve("median_smooth", indices=[], win_length=5),
         "indices must be a non-empty numeric array")


def check_median_quirk(solve):
    """Even windows use the lower median and the output is shorter."""
    r = solve("median_smooth", indices=[1., 5., 2., 9., 3., 7.],
              win_length=4)
    assert r["pad_length"] == 1 and r["shape"] == [4]
    _near(r["smoothed"], [1., 2., 3., 3.])
    r2 = solve("median_smooth",
               indices=[float(i) for i in range(1, 41)], win_length=30)
    assert r2["pad_length"] == 14 and len(r2["smoothed"]) == 25
    _near(r2["smoothed"][0], 1.0)
    _near(r2["smoothed"][-1], 25.0)


# ---------------------------------------------------------------------------
def check_pitch(solve):
    """Full pipeline, frequency ledger and batch pack/unpack."""
    r = solve("pitch", waveform=WF, sample_rate=800, frame_time=0.01,
              freq_low=100, freq_high=800, win_length=3)
    _near(r["shape"], [4])
    _near(r["freq"], [199.99999995] * 4)
    assert r["indices"] == [4, 4, 4, 4, 4]
    assert r["smoothed"] == [4, 4, 4, 4]
    assert r["meta"] == {"lags": 8, "frame_size": 8, "num_frames": 5,
                         "pad": 8, "padded_length": 48}
    _near(r["nccf"][0], NCCF0)
    # 2D batch keeps a leading dimension
    r2 = solve("pitch", waveform=[WF, [v * 2 for v in WF]],
               sample_rate=800, frame_time=0.01, freq_low=100,
               freq_high=800, win_length=3)
    _near(r2["shape"], [2, 4])
    _near(r2["freq"][0], r["freq"])
    _near(r2["freq"][1], r["freq"])
    # default frame_time / win_length / freq_low / freq_high still run
    r3 = solve("pitch", waveform=[0.5] * 360, sample_rate=800)
    _near(r3["shape"], [30])
    assert len(r3["freq"]) == 30
    assert r3["meta"] == {"lags": 10, "frame_size": 8,
                         "num_frames": 45, "pad": 10,
                         "padded_length": 370}
    assert r3["indices"] == [2] * 45
    assert r3["smoothed"] == [2] * 30
    _near(r3["freq"], [399.9999998] * 30, tol=1e-6)


def check_pitch_domain(solve):
    """pitch input contracts."""
    _err(lambda: solve("pitch", waveform=[], sample_rate=800),
         "waveform must be a non-empty numeric array")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       frame_time=-1),
         "frame_time must be a positive number")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       win_length=2),
         "win_length must be at least 3")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       freq_low=2.5),
         "freq_low must be an integer")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       freq_high=0),
         "freq_high must be at least 1")


# ---------------------------------------------------------------------------
# hardening cases added after mutation round 1
# ---------------------------------------------------------------------------
def check_nccf_meta_edges(solve):
    """Non-integral ceil boundaries in lags / frame_size / num_frames."""
    r = solve("nccf", waveform=[1., 2., 3., 4., 5., 6.],
              sample_rate=8, frame_time=0.25, freq_low=3)
    assert (r["lags"], r["frame_size"], r["num_frames"],
            r["pad"], r["padded_length"]) == (3, 2, 3, 3, 9)
    _near(r["shape"], [3, 3])
    _near(r["nccf"][0], [0.12307692289856895, 0.08799999988609039,
                         0.06829268284441538])
    _near(r["nccf"][-1], [0.01366120217774034, 0.0, 0.0])

    r2 = solve("nccf", waveform=[0.5] * 30, sample_rate=1000,
               frame_time=0.0055, freq_low=300)
    assert (r2["lags"], r2["frame_size"], r2["num_frames"],
            r2["pad"], r2["padded_length"]) == (4, 6, 5, 4, 34)
    _near(r2["shape"], [5, 4])
    _near(r2["nccf"][0][0], 0.6666666644893424)
    _near(r2["nccf"][-1][0], 0.6666666643854349)

    # 10 samples / 8-sample frames -> ceil(10/8) = 2, not floor = 1
    r3 = solve("nccf", waveform=[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
               sample_rate=800, frame_time=0.01, freq_low=100)
    assert (r3["lags"], r3["frame_size"], r3["num_frames"],
            r3["pad"], r3["padded_length"]) == (8, 8, 2, 14, 24)
    _near(r3["shape"], [2, 8])
    _near(r3["nccf"][0], [0.004142502070179343, 0.0035603715161640253,
                          0.002959674435090305, 0.0024164595409682265,
                          0.0019310754600042157, 0.0015006002397108783,
                          0.0011204481789716485, 0.0007853970315141392])
    _near(r3["nccf"][1], [0.004972375688874073, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0])


def check_domain_shapes(solve):
    """Scalar / bool / missing-key / bool-flag input contracts."""
    _err(lambda: solve("nccf", waveform=5, sample_rate=800,
                       frame_time=0.01, freq_low=100),
         "waveform must be an array")
    _err(lambda: solve("nccf", waveform=[True] * 8, sample_rate=800,
                       frame_time=0.01, freq_low=100),
         "waveform must be a non-empty numeric array")
    _err(lambda: solve("nccf", waveform=WF, sample_rate=800,
                       frame_time=True, freq_low=100),
         "frame_time must be a positive number")
    # waveform validation must happen before sample_rate is even read
    _err(lambda: solve("nccf", waveform=[]),
         "waveform must be a non-empty numeric array")


def check_combine_max_guard_edges(solve):
    """Every one of the four arrays is validated, not just a_values."""
    _err(lambda: solve("combine_max", a_values=[1.], a_indices=[1],
                       b_values=["x"], b_indices=[1]),
         "b_values must be a non-empty numeric array")
    _err(lambda: solve("combine_max", a_values=[1.], a_indices=[1],
                       b_values=[1.], b_indices=[False]),
         "b_indices must be a non-empty numeric array")


def check_find_max_half_window(solve):
    """Odd lag dimension: half_size floors; 0.999 is NOT close enough."""
    r = solve("find_max", nccf=[[0., 0.1, 0.2, 0.999, 0.1, 0.1, 1.0]],
              sample_rate=800, freq_high=800)
    assert r["half_size"] == 3 and r["lag_min"] == 1
    assert r["indices"] == [7]
    _near(r["best_values"], [1.0])
    assert r["best_indices"] == [5]
    _near(r["half_values"], [0.2])
    assert r["half_indices"] == [1]


def check_find_max_boundaries(solve):
    """Strict > at exactly 0.99 * best, and the 0.98 / 0.99 gap."""
    r = solve("find_max", nccf=[[0., 0.1, 0.99, 0.1, 1.0, 0.1, 0.1, 0.1]],
              sample_rate=800, freq_high=800)
    assert r["indices"] == [5] and r["half_indices"] == [1]
    _near(r["half_values"], [0.99])
    r2 = solve("find_max", nccf=[[0., 0.1, 0.985, 0.1, 1.0, 0.1, 0.1, 0.1]],
               sample_rate=800, freq_high=800)
    assert r2["indices"] == [5] and r2["half_indices"] == [1]
    # 12 lags with a non-integral 800 / 300 -> lag_min = 3, not 2
    row = [0., 0.1, 1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15]
    r3 = solve("find_max", nccf=[row], sample_rate=800, freq_high=300)
    assert r3["lag_min"] == 3 and r3["half_size"] == 6
    assert r3["indices"] == [11]
    assert r3["best_indices"] == [7] and r3["half_indices"] == [2]
    _near(r3["best_values"], [0.9])
    _near(r3["half_values"], [0.4])
    # all-negative rows still reduce to the first maximum
    r4 = solve("find_max", nccf=[[-3., -1., -2., -5., -4., -6., -7.]],
               sample_rate=800, freq_high=800)
    assert r4["indices"] == [2]
    _near(r4["best_values"], [-1.0])


def check_find_max_guard_edges(solve):
    """find_max has its own sample_rate / freq_high lower bounds."""
    _err(lambda: solve("find_max", nccf=[[0.] * 12], sample_rate=0,
                       freq_high=800),
         "sample_rate must be at least 1")
    _err(lambda: solve("find_max", nccf=[[0.] * 12], sample_rate=800,
                       freq_high=0),
         "freq_high must be at least 1")


def check_median_smooth_guard_edges(solve):
    """Non-numeric indices are rejected before any sorting happens."""
    _err(lambda: solve("median_smooth", indices=["a", "b", "c", "d"],
                       win_length=3),
         "indices must be a non-empty numeric array")


def check_pitch_guard_edges(solve):
    """Guard ORDER in the full pipeline: earliest violation wins."""
    _err(lambda: solve("pitch", waveform=[], sample_rate=800, win_length=2),
         "waveform must be a non-empty numeric array")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=0, win_length=2),
         "sample_rate must be at least 1")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       frame_time=-1, win_length=2),
         "frame_time must be a positive number")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       win_length=2, freq_low=2.5),
         "win_length must be at least 3")
    _err(lambda: solve("pitch", waveform=WF, sample_rate=800,
                       freq_low=0, freq_high=0),
         "freq_low must be at least 1")


def check_pitch_freq_high(solve):
    """freq_high must be passed through unchanged (12 Hz / fh=3 case)."""
    wf = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    r = solve("pitch", waveform=wf, sample_rate=12, frame_time=1 / 12,
              freq_low=1, freq_high=3, win_length=3)
    _near(r["shape"], [11])
    _near(r["freq"], [2.39999999952] * 11, tol=1e-6)
    assert r["indices"] == [5] * 12
    assert r["smoothed"] == [5] * 11
    assert r["meta"] == {"lags": 12, "frame_size": 1, "num_frames": 12,
                         "pad": 12, "padded_length": 24}
