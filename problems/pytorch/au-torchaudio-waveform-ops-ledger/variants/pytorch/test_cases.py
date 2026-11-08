#!/usr/bin/env python3
"""857 judge cases: torchaudio functional waveform-operations ledger.

Every check takes `solve` and asserts on explicit, hand-verified values taken
from the verbatim official implementation (see notes.md).  No torch, no import
from the solution module, no global RNG.
"""


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
def check_mu_law(solve):
    """mu=qc-1, log1p companding, +0.5 truncation, decode formula."""
    r = solve("mu_law", x=[-1., -0.5, 0., 0.5, 1.],
              quantization_channels=256)
    _near(r["encoded"], [0, 16, 128, 239, 255])
    assert r["mu"] == 255.0 and r["quantization_channels"] == 256
    _near(r["shape"], [5])
    _near(r["decoded"][0], -1.0)
    _near(r["decoded"][-1], 1.0)
    assert 0 < r["decoded"][2] < 1e-3
    assert all(r["decoded"][i] <= r["decoded"][i + 1] for i in range(4))
    _near(r["decoded"][1], -0.4966766264665899)
    _near(r["decoded"][3], 0.4966766264665899)

    for qc in (2, 4, 8, 16):
        r = solve("mu_law", x=[-1., 0., 1.], quantization_channels=qc)
        _near(r["encoded"], [0, qc // 2, qc - 1])
        assert r["mu"] == float(qc - 1)
    # +0.5 before truncation is round-half-up on the non-negative encoded value
    r = solve("mu_law", x=[0.3], quantization_channels=256)
    _near(r["encoded"], [228])
    _near(r["decoded"], [0.30633397981616217])
    # 2D shape preserved
    r = solve("mu_law", x=[[0.25, -0.25], [0.75, -0.75]],
              quantization_channels=16)
    _near(r["shape"], [2, 2])
    assert all(isinstance(v, int) for row in r["encoded"] for v in row)


def check_mu_law_domain(solve):
    """mu-law input contracts."""
    for bad in (2.5, "8", True):
        _err(lambda b=bad: solve("mu_law", x=[0.1],
                                 quantization_channels=b),
             "quantization_channels must be an integer")
    for bad in (1, 0, -4):
        _err(lambda b=bad: solve("mu_law", x=[0.1],
                                 quantization_channels=b),
             "quantization_channels must be at least 2")
    _err(lambda: solve("mu_law", x=[1.5], quantization_channels=8),
         "x must be scaled to between -1 and 1")
    _err(lambda: solve("mu_law", x=[-1.2], quantization_channels=8),
         "x must be scaled to between -1 and 1")
    _err(lambda: solve("mu_law", x=[], quantization_channels=8),
         "x must be a non-empty numeric array")
    _err(lambda: solve("mu_law", x=["a"], quantization_channels=8),
         "x must be a non-empty numeric array")
    _err(lambda: solve("mu_law", x=0.5, quantization_channels=8),
         "x must be an array")


# ---------------------------------------------------------------------------
def check_convolve(solve):
    """Direct method: true convolution, original-length slicing, swap."""
    r = solve("convolve", x=[1., 2., 3.], y=[1., 1.])
    _near(r["out"], [1., 3., 5., 3.])
    _near(r["full"], [1., 3., 5., 3.])
    assert r["full_length"] == 4 and not r["swapped"]
    _near(r["shape"], [4])
    assert r["x_length"] == 3 and r["y_length"] == 2
    _near(r["leading_shape"], [])
    assert r["num_signals"] == 1

    r = solve("convolve", x=[1., 2., 3., 4.], y=[1., -1.])
    _near(r["out"], [1., 1., 1., 1., -4.])

    # x shorter: operands are swapped but slicing keeps the ORIGINAL lengths
    r = solve("convolve", x=[1., 2.], y=[1., 2., 3.], conv_mode="same")
    assert r["swapped"] is True
    assert r["x_length"] == 2 and r["y_length"] == 3
    _near(r["out"], [4., 7.])
    _near(r["full"], [1., 4., 7., 6.])
    r = solve("convolve", x=[1., 2.], y=[1., 2., 3.], conv_mode="valid")
    _near(r["out"], [4., 7.])

    for mode, exp in (("valid", [1., 1., 1.]), ("same", [1., 1., 1., 1.])):
        r = solve("convolve", x=[1., 2., 3., 4.], y=[1., -1.],
                  conv_mode=mode)
        _near(r["out"], exp)
        assert r["mode"] == mode

    # 2D with paired leading rows
    r = solve("convolve", x=[[1., 2., 3.], [4., 5., 6.]],
              y=[[1., 1.], [2., 2.]])
    _near(r["out"], [[1., 3., 5., 3.], [8., 18., 22., 12.]])
    _near(r["leading_shape"], [2])
    assert r["num_signals"] == 2
    # singleton leading dimension broadcasts
    r = solve("convolve", x=[[1., 2., 3.], [4., 5., 6.]],
              y=[[1., 1.]])
    _near(r["out"], [[1., 3., 5., 3.], [4., 9., 11., 6.]])


def check_convolve_domain(solve):
    """Direct-convolution input contracts."""
    _err(lambda: solve("convolve", x=[[1., 2.]], y=[1., 2.]),
         "The operands must be the same dimension")
    _err(lambda: solve("convolve",
                       x=[[1., 2.], [3., 4.], [5., 6.]],
                       y=[[1., 1.], [2., 2.]]),
         "Leading dimensions of x and y are not broadcastable")
    _err(lambda: solve("convolve", x=[1., 2.], y=[1., 2.],
                       conv_mode="nope"),
         "Unrecognized mode value")
    _err(lambda: solve("convolve", x=[], y=[1., 2.]),
         "x must be a non-empty numeric array")
    _err(lambda: solve("convolve", x=[1., 2.], y=[]),
         "y must be a non-empty numeric array")


# ---------------------------------------------------------------------------
def check_fftconvolve(solve):
    """FFT method: DFT product, no operand swap, same slicing rules."""
    r = solve("fftconvolve", x=[1., 2., 3.], y=[1., 1.])
    _near(r["out"], [1., 3., 5., 3.])
    assert all(isinstance(v, float) for v in r["out"])
    assert r["swapped"] is False
    _near(r["full"], [1., 3., 5., 3.])
    r = solve("fftconvolve", x=[1., 2., 3., 4.], y=[1., -1.])
    _near(r["out"], [1., 1., 1., 1., -4.])
    for mode, exp in (("valid", [1., 1., 1.]), ("same", [1., 1., 1., 1.])):
        r = solve("fftconvolve", x=[1., 2., 3., 4.], y=[1., -1.],
                  conv_mode=mode)
        _near(r["out"], exp)
    r = solve("fftconvolve", x=[1., 2.], y=[1., 2., 3.], conv_mode="same")
    _near(r["out"], [4., 7.])
    assert r["swapped"] is False
    r = solve("fftconvolve", x=[0.5, -0.25, 0.75], y=[0.3, 0.2, 0.1, 0.4])
    _near(r["out"], [0.15, 0.025, 0.225, 0.325, -0.025, 0.3])
    r = solve("fftconvolve", x=[[1., 2., 3.], [4., 5., 6.]],
              y=[[1., 1.]])
    _near(r["out"], [[1., 3., 5., 3.], [4., 9., 11., 6.]])


def check_fftconvolve_domain(solve):
    """FFT-convolution shares the shape/mode contracts."""
    _err(lambda: solve("fftconvolve", x=[[1., 2.]], y=[1., 2.]),
         "The operands must be the same dimension")
    _err(lambda: solve("fftconvolve", x=[1., 2.], y=[1., 2.],
                       conv_mode="bad"),
         "Unrecognized mode value")
    _err(lambda: solve("fftconvolve", x=[1., 2.], y=["a"]),
         "y must be a non-empty numeric array")


# ---------------------------------------------------------------------------
def check_slice_mode(solve):
    """_apply_convolve_mode center cropping with floor division."""
    f = [1., 2., 3., 4., 5.]
    r = solve("slice_mode", full=f, x_length=3, y_length=3,
              conv_mode="full")
    _near(r["out"], f)
    assert r["start_idx"] == 0 and r["target_length"] == 5
    r = solve("slice_mode", full=f, x_length=3, y_length=3,
              conv_mode="valid")
    _near(r["out"], [3.])
    assert r["start_idx"] == 2 and r["target_length"] == 1
    r = solve("slice_mode", full=f, x_length=3, y_length=3,
              conv_mode="same")
    _near(r["out"], [2., 3., 4.])
    assert r["start_idx"] == 1 and r["target_length"] == 3
    # valid length uses max-min+1, not |x-y|
    r = solve("slice_mode", full=[1., 2., 3., 4.], x_length=2,
              y_length=3, conv_mode="valid")
    _near(r["out"], [2., 3.])
    assert r["start_idx"] == 1 and r["target_length"] == 2
    # same length is x_length even when x is the shorter operand
    r = solve("slice_mode", full=[1., 2., 3., 4.], x_length=2,
              y_length=3, conv_mode="same")
    _near(r["out"], [2., 3.])
    assert r["target_length"] == 2
    # odd full length: floor centering biases left
    r = solve("slice_mode", full=[1., 2., 3., 4., 5., 6., 7.],
              x_length=4, y_length=4, conv_mode="same")
    _near(r["out"], [2., 3., 4., 5.])
    assert r["start_idx"] == 1
    r = solve("slice_mode", full=[[1., 2., 3.], [4., 5., 6.]],
              x_length=2, y_length=2, conv_mode="same")
    _near(r["out"], [[1., 2.], [4., 5.]])


def check_slice_mode_domain(solve):
    """Slicing-mode input contracts and the unknown-mode dispatch."""
    _err(lambda: solve("slice_mode", full=[1., 2., 3.], x_length=2,
                       y_length=2, conv_mode="nope"),
         "Unrecognized mode value")
    _err(lambda: solve("slice_mode", full=[1., 2., 3.], x_length=0,
                       y_length=2),
         "x_length must be at least 1")
    _err(lambda: solve("slice_mode", full=[1., 2., 3.], x_length=2.5,
                       y_length=2),
         "x_length must be an integer")
    _err(lambda: solve("slice_mode", full=[1., 2., 3.], x_length=2,
                       y_length=1.5),
         "y_length must be an integer")
    _err(lambda: solve("slice_mode", full=[], x_length=2, y_length=2),
         "full must be a non-empty numeric array")
    _err(lambda: solve("nope", full=[1., 2., 3.], x_length=2, y_length=2),
         "unknown mode")


# ---------------------------------------------------------------------------
def check_add_noise(solve):
    """SNR scaling, lengths mask, and the unmasked-noise quirk."""
    r = solve("add_noise", waveform=[1., 2., 3., 4.],
              noise=[1., 1., 1., 1.], snr=0.0)
    _near(r["energy_signal"], 30.0)
    _near(r["energy_noise"], 4.0)
    _near(r["scale"], 2.73861278752583)
    _near(r["out"], [3.73861278752583, 4.73861278752583,
                     5.73861278752583, 6.73861278752583])
    _near(r["shape"], [4])
    _near(r["leading_shape"], [])
    assert r["length"] == 4
    r = solve("add_noise", waveform=[1., 2., 3., 4.],
              noise=[1., 1., 1., 1.], snr=10.0)
    _near(r["scale"], 0.8660254037844385)
    _near(r["out"], [1.8660254037844384, 2.8660254037844384,
                     3.8660254037844384, 4.866025403784438])
    # per-row snr vector
    r = solve("add_noise", waveform=[[1., 2., 3.], [4., 5., 6.]],
              noise=[[1., 0., 1.], [0., 1., 0.]], snr=[0., 10.])
    _near(r["energy_signal"], [14., 77.])
    _near(r["energy_noise"], [2., 1.])
    _near(r["scale"], [2.6457513110645903, 2.7748873851023217])
    _near(r["leading_shape"], [2])
    _near(r["out"], [[3.6457513110645903, 2., 5.64575131106459],
                     [4., 7.774887385102321, 6.]])
    # lengths masks the energy but NOT the added noise
    r = solve("add_noise", waveform=[1., 2., 3., 100.],
              noise=[1., 1., 1., 1.], snr=0.0, lengths=3)
    _near(r["energy_signal"], 14.0)
    _near(r["energy_noise"], 3.0)
    _near(r["scale"], 2.1602468994692865)
    _near(r["out"][3], 102.16024689946929)
    # negative SNR gives a larger scale
    r = solve("add_noise", waveform=[1., 2.], noise=[1., 1.], snr=-6.0)
    assert r["scale"] > solve("add_noise", waveform=[1., 2.],
                              noise=[1., 1.], snr=0.0)["scale"]


def check_add_noise_domain(solve):
    """add_noise rank / length / energy contracts."""
    _err(lambda: solve("add_noise", waveform=[[1., 2.], [3., 4.]],
                       noise=[[1., 1.], [1., 1.]], snr=0.0),
         "Input leading dimensions don't match.")
    _err(lambda: solve("add_noise", waveform=[1., 2., 3.],
                       noise=[1., 1.], snr=0.0),
         "Length dimensions of waveform and noise don't match")
    _err(lambda: solve("add_noise", waveform=[[1., 2.]],
                       noise=[[1., 1.]], snr=[0.],
                       lengths=[[2], [2]]),
         "Input leading dimensions don't match.")
    _err(lambda: solve("add_noise", waveform=[0., 0.],
                       noise=[1., 1.], snr=0.0),
         "energy_signal and energy_noise must be positive")
    _err(lambda: solve("add_noise", waveform=[1., 2.],
                       noise=[0., 0.], snr=0.0),
         "energy_signal and energy_noise must be positive")
    _err(lambda: solve("add_noise", waveform=[], noise=[1.], snr=0.0),
         "waveform must be a non-empty numeric array")


# ---------------------------------------------------------------------------
def check_phase_vocoder(solve):
    """Phase accumulation, magnitude interpolation, rate short-circuit."""
    c = {"real": [[1., 2., 3.], [1., 1., 1.], [0., 0., 0.]],
         "imag": [[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]],
         "rate": 1.5, "phase_advance": [[0.], [1.], [2.]]}
    r = solve("phase_vocoder", **c)
    _near(r["shape"], [3, 2])
    _near(r["real"], [[1., 2.5], [1., 1.], [0., 0.]])
    _near(r["imag"], [[0., 0.], [1., 1.], [1., 1.]])
    _near(r["time_steps"], [0., 1.5])
    _near(r["alphas"], [0., 0.5])
    assert r["num_frames_in"] == 3 and r["num_frames_out"] == 2
    assert r["freq"] == 3 and r["rate"] == 1.5

    # rate == 1.0 returns the input verbatim (no polar round-trip)
    r = solve("phase_vocoder", **dict(c, rate=1.0))
    assert r["real"] == c["real"] and r["imag"] == c["imag"]
    _near(r["shape"], [3, 3])
    _near(r["time_steps"], [0., 1., 2.])
    assert r["num_frames_out"] == 3

    for rate, ts, frames in ((2.0, [0., 2.], 2),
                             (0.75, [0., 0.75, 1.5, 2.25], 4),
                             (3.0, [0.], 1), (1.3, [0., 1.3, 2.6], 3)):
        r = solve("phase_vocoder", **dict(c, rate=rate))
        _near(r["time_steps"], ts)
        _near(r["alphas"], [t % 1.0 for t in ts])
        assert r["num_frames_out"] == frames
        assert len(r["shape"]) == 2 and r["shape"][-1] == frames
    r = solve("phase_vocoder", **dict(c, rate=0.75))
    _near(r["real"][0], [1., 1.75, 2.5, 2.25])
    # batched leading dimension
    cb = {"real": [[[1., 2., 3.], [1., 1., 1.], [0., 0., 0.]]],
          "imag": [[[0., 0., 0.], [1., 1., 1.], [1., 1., 1.]]],
          "rate": 1.5, "phase_advance": [[0.], [1.], [2.]]}
    r = solve("phase_vocoder", **cb)
    _near(r["shape"], [1, 3, 2])
    _near(r["real"][0][0], [1., 2.5])
    # non-trivial phase_advance and phases
    r = solve("phase_vocoder",
              real=[[2., 1., 0.5, 0.25], [0., 0.5, 1., 1.5]],
              imag=[[1., 1., 1., 1.], [-1., -0.5, 0., 0.5]],
              rate=1.3, phase_advance=[[0.5], [1.5]])
    _near(r["shape"], [2, 4])
    _near(r["time_steps"], [0., 1.3, 2.6, 3.9])
    assert r["freq"] == 2 and r["num_frames_in"] == 4
    _near(r["real"], [[2.0, 0.9371708245126286, 0.4765863337187867,
                      0.0249999999999999],
                     [6.123233995736766e-17, 0.5621320343559644,
                      1.348683298050514, 0.14999999999999947]])
    _near(r["imag"], [[1.0, 0.9371708245126285, 0.9531726674375732,
                      0.09999999999999964],
                     [-1.0, -0.5621320343559643, 0.0,
                      0.049999999999999815]])


def check_phase_vocoder_domain(solve):
    """phase_vocoder input contracts."""
    base = {"real": [[1., 2., 3.]], "imag": [[0., 0., 0.]],
            "rate": 1.5, "phase_advance": [[0.]]}
    _err(lambda: solve("phase_vocoder", **dict(base, imag=[[0., 0.]])),
         "real and imag must have the same shape")
    for bad in (0, -1, 0.0, "2", None):
        _err(lambda b=bad: solve("phase_vocoder", **dict(base, rate=b)),
             "rate must be a positive number")
    _err(lambda: solve("phase_vocoder",
                       **dict(base, real=[1., 2., 3.], imag=[0., 0., 0.])),
         "complex_specgrams must have (..., freq, frames) shape")
    _err(lambda: solve("phase_vocoder", **dict(base,
                       phase_advance=[[0.], [1.]])),
         "phase_advance must have one value per freq bin")
    _err(lambda: solve("phase_vocoder", **dict(base, real=[[]], imag=[[]])),
         "real must be a non-empty numeric array")
    _err(lambda: solve("phase_vocoder", **dict(base,
                       phase_advance=["a"])),
         "phase_advance must be a non-empty numeric array")


# ---------------------------------------------------------------------------
# hardening cases added after mutation round 1
# ---------------------------------------------------------------------------
def check_convolve_broadcast_nd(solve):
    """Asymmetric multi-dimensional leading broadcast keeps axis order."""
    x = [[[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]],
         [[2., 1., 0., -1.], [1., 0., -1., -2.], [0., -1., -2., -3.]]]
    y = [[[1., 1., 1., 1.]], [[1., -1., 1., -1.]]]
    r = solve("convolve", x=x, y=y)
    _near(r["leading_shape"], [2, 3])
    assert r["num_signals"] == 6
    _near(r["shape"], [2, 3, 7])
    _near(r["out"][0][0], [1., 3., 6., 10., 9., 7., 4.])
    _near(r["out"][0][2], [3., 7., 12., 18., 15., 11., 6.])
    _near(r["out"][1][2], [0., -1., -1., -2., 2., -1., 3.])


def check_convolve_singleton_values(solve):
    """A singleton leading row must broadcast every last-dim value."""
    r = solve("convolve", x=[[1., 2., 3.], [4., 5., 6.]], y=[[1., 2.]])
    _near(r["out"], [[1., 4., 7., 6.], [4., 13., 16., 12.]])
    _near(r["leading_shape"], [2])
    assert r["num_signals"] == 2


def check_convolve_equal_length(solve):
    """Equal-length operands are not swapped, and a 3-tap kernel is real conv."""
    r = solve("convolve", x=[1., 2., 3.], y=[1., 2., 3.])
    _near(r["out"], [1., 4., 10., 12., 9.])
    assert r["swapped"] is False
    assert r["x_length"] == 3 and r["y_length"] == 3
    assert r["full_length"] == 5


def check_slice_mode_offset_2d(solve):
    """A positive start index must survive the recursive last-axis slice."""
    r = solve("slice_mode",
              full=[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]],
              x_length=3, y_length=3, conv_mode="valid")
    _near(r["out"], [[3.], [8.]])
    assert r["start_idx"] == 2 and r["target_length"] == 1


def check_slice_mode_unit_ylength(solve):
    """y_length == 1 is legal; valid length is max-min+1."""
    r = solve("slice_mode", full=[1., 2., 3.], x_length=3, y_length=1,
              conv_mode="valid")
    _near(r["out"], [1., 2., 3.])
    assert r["start_idx"] == 0 and r["target_length"] == 3


def check_domain_guard_order(solve):
    """Validation order is part of the contract when two inputs are bad."""
    _err(lambda: solve("convolve", x=[], y=[]),
         "x must be a non-empty numeric array")
    _err(lambda: solve("convolve", x=[[1., 2.]], y=[1., 2.],
                       conv_mode="nope"),
         "The operands must be the same dimension")
    _err(lambda: solve("add_noise", waveform=[], noise=[], snr=0.0),
         "waveform must be a non-empty numeric array")


def check_add_noise_broadcast_conflict(solve):
    """Conflicting snr leading dimensions are rejected, not silently grown."""
    _err(lambda: solve("add_noise",
                       waveform=[[1., 2., 3.], [4., 5., 6.]],
                       noise=[[1., 1., 1.], [1., 1., 1.]],
                       snr=[0., 10., 20.]),
         "shapes are not broadcastable")


def check_bool_array_elements(solve):
    """bool is not a numeric sample (isinstance(True, int) trap)."""
    _err(lambda: solve("mu_law", x=[True], quantization_channels=8),
         "x must be a non-empty numeric array")
    _err(lambda: solve("convolve", x=[1., True], y=[1., 2.]),
         "x must be a non-empty numeric array")
