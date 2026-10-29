# TTFT / TPOT Timeline for Continuous Batching: Prefill and Decode Share One Step

## Problem

Online inference services are judged by two SLO metrics:

- **TTFT** (time to first token): the latency from a request's arrival to its first emitted token — how fast the user perceives the service;
- **TPOT / ITL** (time per output token / inter-token latency): the gap between consecutive tokens after the first — how smoothly the answer streams.

Neither is a compute problem; both are **scheduling** problems. Continuous batching packs the decode of many requests into one forward pass, which lifts throughput — but prefill and decode compete for that same forward pass. Every prefill step is a step in which all decoding requests emit nothing. A newcomer's prefill therefore stretches the running requests' TPOT, which is exactly the trade-off chunked prefill has to balance.

This problem simulates a minimal serving engine in integer steps: given arrival times, prompt lengths, generation lengths, a prefill chunk size and a batch limit, output each request's TTFT, decode span and finish step, plus the total prefill / decode step counts.

## Rules

Time is measured in integer steps; step 0 is the first time point. An **active request** is one that has been admitted and not yet finished (both prefilling and decoding hold one batch slot).

Each step:

1. **Admission**: while the number of active requests is below `max_batch`, admit waiting requests that have already arrived, in `(arrivals, index)` ascending order, until the batch is full or no arrived request remains. A request admitted in this step may take part in this step's action.
2. **Action** (exactly one per step):
   - **prefill step**: if some active request still has prompt tokens left, take the earliest admitted one (FCFS) and advance it by `min(chunk, remaining)` prompt tokens. No decoding happens this step: no running request emits anything;
   - **decode step**: otherwise, if at least one active request has finished prefilling, every prefill-complete, unfinished active request emits exactly one token. A request that reaches `gen_lens[i]` tokens finishes in this step and releases its slot; a slot released this step can only be reused by the admission phase of the next step;
   - **idle**: if neither applies, jump the clock straight to the next arrival step; idle steps are not counted anywhere.

The step that completes a prefill emits **no** token; the first token appears in the following decode step.

Return a five-tuple:

- `ttft`: for each request, "first-token step − arrival step";
- `decode_span`: for each request, "last-token step − first-token step", 0 when `gen_lens[i] == 1`;
- `finish`: for each request, the absolute step of its last token;
- `prefill_steps`: how many prefill steps were performed overall;
- `decode_steps`: how many decode steps were performed overall.

## Examples

**Example 1**: `arrivals = [0, 0, 2]`, `prompt_lens = [8, 4, 4]`, `gen_lens = [2, 3, 1]`, `chunk = 4`, `max_batch = 2`

```
step 0  admit r0, r1 (r2 not yet)   prefill r0: 8 -> 4
step 1  prefill r0: 4 -> 0 (done)
step 2  prefill r1: 4 -> 0 (done)   (r2 arrived at step 2, but the batch is full)
step 3  decode: r0 token 1, r1 token 1   <- first token of both
step 4  decode: r0 token 2 -> done; r1 token 2
step 5  admit r2 (takes r0's slot)  prefill r2: 4 -> 0
step 6  decode: r1 token 3 -> done; r2 token 1 -> done (gen_len = 1)
```

```
ttft        = [3, 3, 4]     # r2 only arrived at step 2 and was admitted at step 5
decode_span = [1, 3, 0]
finish      = [4, 6, 6]
prefill_steps = 4
decode_steps  = 3
```

**Example 2** (prefill crowds out decode): `arrivals = [0, 2]`, `prompt_lens = [1, 8]`, `gen_lens = [4, 1]`, `chunk = 2`, `max_batch = 2`

```
step 0  admit r0                 prefill r0: 1 -> 0
step 1  decode r0 token 1        <- r0's TTFT is only 1 step
step 2  admit r1                 prefill r1: 8 -> 6
step 3  prefill r1: 6 -> 4
step 4  prefill r1: 4 -> 2
step 5  prefill r1: 2 -> 0 (done)
step 6  decode r0 token 2, r1 token 1
step 7  decode r0 token 3
step 8  decode r0 token 4 -> done
```

```
ttft        = [1, 4]
decode_span = [7, 0]        # r0's 3 gaps average 7/3 steps instead of 1
finish      = [8, 6]
prefill_steps = 5
decode_steps  = 4
```

**Example 3** (one slot, strictly serial): `arrivals = [0, 0, 0]`, `prompt_lens = [1, 1, 1]`, `gen_lens = [1, 1, 1]`, `chunk = 1`, `max_batch = 1`

```
ttft        = [1, 3, 5]
decode_span = [0, 0, 0]
finish      = [1, 3, 5]
prefill_steps = 3
decode_steps  = 3
```

## Constraints

- At most 64 requests; `arrivals[i] >= 0` and may repeat or be out of order; `prompt_lens[i] >= 1`; `gen_lens[i] >= 1`; `chunk >= 1`; `max_batch >= 1`.
- Every request eventually finishes (`max_batch >= 1` guarantees the waiting queue cannot deadlock).
- The two lists and the two counts are Python `int`s; `finish` holds absolute step numbers, not latencies relative to arrival.
- Only the numbers are judged; any simulation style is fine (dict + single loop, event queue, timing wheel).
