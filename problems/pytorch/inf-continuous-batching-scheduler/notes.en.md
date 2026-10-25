# Solution Notes · Continuous Batching Scheduler

## Contrast with static batching

Traditional static batching requires all requests in a batch to finish at the same time — short requests must wait for long ones, and slots idle. Continuous batching lets requests **join and leave** the running set at any time, pushing GPU utilization close to saturation.

## Prefill simplification

In real systems, prefill is "processing prompt_len tokens in one step"; this problem abstracts it as 1 extra step (equivalent to treating prefill as one special decode token). This does not change the scheduling semantics; it just makes the total step count easy to compute.

## Admit timing

Each step **admits first, then consumes**. So a request with `arrival=t, gen_len=2`: at step t it is admitted and consumed → rem=2; at t+1 rem=1; at t+2 rem=0, finish. It occupies 3 steps (1 prefill + 2 decode).

## FCFS ordering

Ascending by `(arrival_step, request_id)`, guaranteeing "same arrival → input order". This was also the default policy of early vLLM versions.
