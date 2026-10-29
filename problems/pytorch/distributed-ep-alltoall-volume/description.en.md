# Distributed: MoE EP all-to-all Communication Volume and Straggler Accounting

## Description

In Expert Parallelism (EP), each rank routes tokens to the rank holding their target expert. `routing[r, e]` is the number of tokens source rank \(r\) routes to expert \(e\). Expert placement is fixed at rank \(e\bmod W\); the expert count need not be divisible by the rank count.

Build a ledger for **one dispatch all-to-all**: routing counts, per-rank network send+receive bytes, imbalance, and a bandwidth-only time estimate. All supplied routing counts are honored; token dropping is not modeled.

## Communication and timing rules

The send-count matrix is

\[
S_{rq}=\sum_{e:\ e\bmod W=q}\text{routing}[r,e],
\qquad R=S^\mathsf{T}.
\]

Both \(S\) and the receive matrix \(R\) **retain diagonal entries**, but diagonal routes target local experts and do not traverse the network. With \(b\) bytes per token, rank \(r\)'s aggregate network send+receive bytes are

\[
B_r=\left(\sum_q S_{rq}+\sum_q S_{qr}-2S_{rr}\right)b.
\]

The local entry appears once in the row sum and once in the column sum, so subtract it twice. A cross-rank transfer contributes load at both its sender and its receiver.

Let \(M=\max_r B_r\) and \(\mu=\frac{1}{W}\sum_r B_r\). Imbalance is \(M/\mu-1\), or 0 when all network loads are zero.

`bandwidth` is each rank's **aggregate network send+receive bandwidth in bytes/second**, so estimated time is \(M/\text{bandwidth}\). This is not a full-duplex `max(send, receive)` model, nor the total time for dispatch plus reverse combine. Ignore computation latency, communication startup latency, metadata, and reverse combine. Full types and valid-input constraints belong to the starter.

## Example

```text
routing=[[1, 2, 3], [4, 5, 6]]
bytes_per_token=4, bandwidth=1e9
send_counts=[[4, 2], [10, 5]]
recv_counts=[[4, 10], [2, 5]]
per_rank_bytes=[48, 48]
max_load_bytes=48, imbalance=0.0, wall_clock=4.8e-8 seconds
```

Experts 0 and 2 belong to rank 0; expert 1 belongs to rank 1. The two cross-rank routes contain 2 and 10 tokens, so each rank handles \((2+10)\times4=48\) network bytes. The 4 and 5 local tokens contribute no network load. With the first three output tensors displayed as lists:

```text
([[4, 2], [10, 5]], [[4, 10], [2, 5]], [48, 48], 48, 0.0, 4.8e-8)
```

A single rank, all-local routing, or all-zero routing produces zero network bytes and time, even when the send-count diagonal is nonzero.
