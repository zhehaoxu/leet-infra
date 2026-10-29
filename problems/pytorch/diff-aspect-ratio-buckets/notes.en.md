# Solution Notes · Aspect-Ratio Buckets and Nearest-Bucket Assignment

## Source

- `huggingface/diffusers`, `src/diffusers/trainers/training_utils.py`: `generate_aspect_ratio_buckets(resolution, divisibility=16, base_resolutions=None)` together with the default table `_DEFAULT_BUCKET_ASPECT_RATIOS`; `find_nearest_bucket(h, w, bucket_options)` scores with `abs(h * bucket_w - w * bucket_h)` and compares with `<=`. The docstring notes that `divisibility=16` satisfies the Flux / Z-Image VAE patch-size constraints.
- This problem makes "generate buckets + assign" a pure function; `parse_buckets_string` in the same file is config parsing and is out of scope.

## Core Points and Pitfalls

**The two lists use opposite orientations.** Each entry of `_DEFAULT_BUCKET_ASPECT_RATIOS` is treated as `(rw, rh)` (width, height), while a user-supplied `base_resolutions` entry is `(h, w)`; the source first builds `aspect_ratios = [(w, h) for (h, w) in base_resolutions]` before entering the same loop. Treating both as `(h, w)` turns a portrait base such as `(1080, 1920)` into landscape buckets — a 90° flip.

**Height-first versus width-first.** The source computes `h = sqrt(target / aspect)` then `w = h * aspect`; one could instead compute `w = sqrt(target * aspect)` then `h = w / aspect`. They are algebraically equal, but floats can differ by an ulp at `.5` rounding boundaries. The solution follows the source order and the independent oracle uses the other path; on the test grid both must produce the same bucket table.

**Python's `round` is banker's rounding.** `round(2.5) == 2`, `round(3.5) == 4`; `int(x + 0.5)` or `floor(x + 0.5)` shifts systematically at exactly `.5` and changes the ladder.

**`max(divisibility, ...)` is not decoration.** With extreme aspect ratios or low resolution, `round` can yield 0; the floor guarantees at least one granularity step per side. The side effect is that different ratios may collapse into one bucket, so deduplication must preserve order: at low resolution the input `(16, 16)` leaves a single bucket.

**Use the cross-multiplied metric, not a ratio difference.** `|h * bw - w * bh|` is pure integer arithmetic and equals the distance between \(h/w\) and \(bh/bw\) up to a positive factor, with no division; `|h/w - bh/bw|` jitters in floating point for close ratios and may not match the source.

**`<=` lets the later bucket win.** This is not arbitrary: one image can be exactly equidistant from two different ratios (e.g. `1088×976` scores `57344` against both `1:1` and `576:464`), and the source's `<=` deterministically picks the later bucket. Switching to `<` would make the answer implementation-dependent; the problem requires matching the source.

**The output is indices, not buckets.** `assignments` has the same length as `image_sizes`; an empty `(0, 2)` input is legal and returns an empty index tensor. The bucket tensor keeps generation order and must not be sorted.

**The pixel budget is only approximate.** The coarser the granularity and the lower the resolution, the larger the deviation of `h*w` from `resolution²` (over 6% at `div=64`, `res=256`); the docs and tests only assert a 5% bound at `divisibility=16`.

## Division of Labor

- `dt-*` (tokenization/data collation) padding and packing operate on sequence lengths; this problem works on image spatial sizes, a different unit.
- `mm-smart-resize` and `mm-clip-preprocess` resize a single image to a fixed size; this problem keeps the aspect ratio, builds discrete buckets, and aligns shapes without interpolation.
- `diff-sd3-timestep-density` and `diff-min-snr-weight` are training-side data plumbing too, but over timesteps; this one is over spatial resolution.
- The resize/letterbox in `vision-*` detection problems is a per-image geometric transform with coordinate mapping; here only discrete bucket indices are returned.
