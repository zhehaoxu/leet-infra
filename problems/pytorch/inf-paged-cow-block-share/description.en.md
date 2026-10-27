# PagedAttention Page Tables with Copy-on-Write Sharing

## Problem

A paged KV cache chops each sequence's KV into fixed-size blocks (`block_size` tokens) and gives the sequence a page table mapping logical pages to physical blocks instead of one contiguous slab of memory. Physical blocks come from a global pool, and every access goes through the table — two lookups instead of pointer arithmetic.

The page table buys two things:

- **Sharing**: multiple beams, parallel samples, or prefix-cache hits can point at the same physical blocks. Each block keeps a refcount. A fork is just "copy the table, bump every refcount" — no memory moves at all.
- **Copy-on-write**: as soon as one sequence writes into a shared block, it must first copy that block into a private one (decrement the old refcount, set the new one to 1) and write there. Skipping the copy silently corrupts the other sequence's KV.

This problem packs both into one integer state machine and asks for the page tables, the refcounts, and the number of copies.

## Task

Implement `solve(num_blocks, block_size, ops)`:

- Inputs:
  - `num_blocks`: total physical blocks, numbered `0..num_blocks-1`;
  - `block_size`: tokens per block;
  - `ops`: `(K, 3)` int64, each row `(kind, seq, arg)`:
    - `kind=0` create: create sequence `seq` holding `arg` tokens (`arg >= 1`);
    - `kind=1` fork: sequence `arg` copies `seq`'s page table and shares its blocks;
    - `kind=2` append: append `arg` tokens to sequence `seq` (`arg >= 1`);
    - `kind=3` free: drop all block references held by `seq` (`arg` ignored).
- Outputs:
  - `page_tables`: `(S, P)` int64, `S` = max sequence id + 1, `P` = the largest page count any sequence reached, short rows padded with `-1`;
  - `refcount`: `(num_blocks,)` int64 after the whole simulation;
  - `copies`: int64 scalar, how many copy-on-writes happened.
- Allocation: always take the **lowest-numbered** free block; `free` only decrements refcounts, and a block returns to the pool when its count hits 0.
- Sequence ids may not be reused (re-creating a freed id is an error); raise `ValueError` when physical blocks run out or an operation is illegal.

## Example

```
num_blocks = 6, block_size = 4

ops: create seq0 (6 tokens) -> 2 pages -> blocks 0,1     refcount = [1,1,0,0,0,0]
     create seq1 (4 tokens) -> 1 page  -> block 2       refcount = [1,1,1,0,0,0]
     fork   seq0 -> seq2                                refcount = [2,2,1,0,0,0]
     append seq2 (+3)       : page 1 shared -> copy block 3
                              page 2 new    -> block 4   refcount = [2,1,1,1,1,0]  copies = 1
     fork   seq2 -> seq3                                refcount = [3,1,1,2,2,0]
     append seq3 (+1)       : lands in shared page 2 -> copy block 5
                                                        refcount = [3,1,1,2,1,1]  copies = 2

page_tables = [[0,1,-1], [2,-1,-1], [0,3,4], [0,3,5]]
refcount    = [3, 1, 1, 2, 1, 1]
copies      = 2
```

## Constraints

- **A fork copies no data**: it clones the table positionally and increments refcounts; `copies` must not move.
- **Only written shared pages get copied**: decide by which logical pages the token range `[old token count, old token count + arg)` lands on, not by "does this sequence share anything".
- **A page that is exactly full is not written**: if the old token count is a multiple of `block_size`, the first appended token lands on a **new** page, so a fresh block is allocated and no copy-on-write happens.
- **A freed sequence still shows up in the output**: report the final snapshot per sequence; rows of freed sequences keep the table they had at release time.
- **A refcount must reach 0 to be free**: with several sharers, one `free` only decrements.
- Rows are padded with `-1` rather than `0` because block 0 is a legitimate block.
