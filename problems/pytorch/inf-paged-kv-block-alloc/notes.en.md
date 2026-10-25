# Solution Notes · PagedAttention Block Allocation

## Rounding the block count up

A sequence length `len` in $[\,(j-1)b,\; jb\,]$ needs $j$ blocks, i.e. $\lceil \text{len}/b\rceil$. The last block is usually only partly filled (internal fragmentation is squeezed below $b$ tokens), which is exactly what paging saves compared with "reserving the whole contiguous span". The integer form `-(-len//b)` avoids floating-point error.

## Admission order = first come, first served

In real vLLM, block allocation happens whenever a decode step appends a token (a new block is `allocate`d only when the current block is full), and it works with preemption to reclaim blocks from old requests and give them to new ones. This problem simplifies it to "a batch of sequences admitted in one go", greedily taking blocks strictly in request-id order: allocate whenever there are enough free blocks, otherwise reject the whole sequence (no partial allocation), so a later short sequence may still be admitted after a long one fails.

## The meaning of the block table

`block_tables[i]` is exactly the **page table** of sequence $i$: logical block $j$ (the KV of tokens $j\cdot b$ through $(j+1)b$) → physical block number `block_tables[i][j]`. When the attention kernel reads KV it first looks up the page table and then locates the GPU memory, which enables non-contiguous storage and cross-request sharing (prefix caching is precisely about making multiple page tables point to the same physical block).
