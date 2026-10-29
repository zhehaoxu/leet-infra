# Training Data: Parsing and Validating Aspect-Ratio Bucket Strings

## Problem

Multi-resolution training must pick a bucket (a `height, width` size pair) for every image, and the bucket list is usually passed as a command-line string such as `--bucket "1024,1024; 768,1344"`. Before training starts, the script has to turn that config into a list of sizes: malformed input must fail immediately, while sizes that are not aligned to the VAE's 8× downsampling only deserve a warning. This problem implements that parser.

## Rules

The input is `buckets_str`; the output is `[(height, width), ...]` in input order:

- A non-string or an empty string raises `ValueError`; a whitespace-only string yields no valid bucket after splitting and raises `ValueError` too.
- Strip the whole string first, then split on `;`; every segment must match `^\s*(\d+)\s*,\s*(\d+)\s*$` **completely**: only decimal digit characters, whitespace allowed around the comma and at both ends of a segment, and an extra comma, a minus sign, a plus sign, a decimal point, scientific notation or a separator such as `512x768` all raise `ValueError`.
- After converting both parts with `int` they must be positive: `0` is invalid as well. The message for this case looks like `Invalid integer in bucket pair '0,0': Bucket dimensions must be positive integers.`.
- When either edge is not a multiple of 8, emit **exactly one** `warning` (its message contains `not divisible by 8`) and keep the bucket. Warnings are per bucket, so a bucket with both edges misaligned still produces a single one.
- The result keeps input order, duplicate buckets are not merged, and each element is an `(int, int)` tuple.

## Example

```
input:  solve("1024,1024; 768,1344; 1344,768")
output: [(1024, 1024), (768, 1344), (1344, 768)]

input:  solve("  512 , 768 ;  256 ,256 ")
output: [(512, 768), (256, 256)]

input:  solve("512,512;16,16;512,512")
output: [(512, 512), (16, 16), (512, 512)]
```

`"500,1008"` returns `[(500, 1008)]` and emits one warning (500 is not a multiple of 8); `"3,5"` also emits just one warning, because the granularity is the bucket, not the edge.

## Constraints

- The only separator is `;`, and the only in-segment separator is `,`; `|`, newlines and spaces are not separators.
- Newlines at the ends of the string are removed by the initial `strip`, so `"512,768\n"` is valid.
- `None`, numbers and other non-string inputs are rejected.
- Any parse or validation failure raises `ValueError`; a size that is not a multiple of 8 only warns and never raises.
