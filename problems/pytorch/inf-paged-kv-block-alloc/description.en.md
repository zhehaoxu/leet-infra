# PagedAttention: KV Cache Paged Block Allocation

## Problem

Traditional KV cache reserves one whole **contiguous** chunk of GPU memory for each sequence; sequences of varying length cause large amounts of internal fragmentation (reserved but unused) and external fragmentation (free space that cannot be stitched together). **PagedAttention** (vLLM) borrows from OS paging: split the KV cache into fixed-size **blocks** (each holding the K/V of `block_size` tokens), and sequences no longer need contiguous memory but instead hold a **block table** — mapping "logical block $j$" to any "physical block number". This brings memory utilization close to 100% and is the foundation of vLLM's high throughput.

This problem simulates the **admission allocation** during one scheduling round: the physical block pool `free_blocks` is given in available order, and several pending sequences apply in **request-id order** (first come, first served), each needing $\lceil \text{len}/\text{block\_size}\rceil$ physical blocks. If the remaining free blocks cannot satisfy the whole sequence, that sequence **cannot be admitted this step** (no partial allocation) and keeps an empty table.

## Rules

Given `free_blocks` (list[int], physical block ids; allocated from the **head**, i.e. `pop(0)` semantics), `block_size` (int), and `seq_lens` (1-D, the token count of each sequence, in ascending request-id order):

1. For each sequence $s$ (preserving the given order): the requirement is $\mathrm{need} = \lceil s/\text{block\_size}\rceil$, and $\mathrm{need}=0$ when $s=0$
2. If the current number of remaining free blocks $\ge \mathrm{need}$: take `need` block ids from the **head** of the free pool in order as that sequence's block table, and remove them from the free pool
3. Otherwise, that sequence's block table is recorded as `[]` (not admitted), occupying **no** blocks
4. After processing all sequences, return each sequence's block table and the remaining free pool

Return the pair `(block_tables, remaining_free)`: `block_tables` is `list[list[int]]` (length = number of sequences; `[]` for unadmitted ones), and `remaining_free` is `list[int]` (physical blocks not yet occupied, preserving their original relative order).

## Examples

```
free_blocks = [0,1,2,3,4,5], block_size = 4
seq_lens    = [10, 4, 6, 5]   → need = [3, 1, 2, 2]  (8 needed in total > 6)

s=10: need=3 → [0,1,2], free=[3,4,5]
s=4:  need=1 → [3],     free=[4,5]
s=6:  need=2 → [4,5],   free=[]
s=5:  need=2 → not enough free → [] (not admitted)

block_tables    = [[0,1,2],[3],[4,5],[]]
remaining_free  = []
```

## Constraints

- CPU; `block_size >= 1`; physical block ids are distinct.
- Allocation strictly follows the given order of `seq_lens`, and each sequence takes blocks from the head of the free pool.
- Unadmitted sequences are represented by an empty list `[]` (no partial allocation, and they do not block later short sequences that can still be admitted).
