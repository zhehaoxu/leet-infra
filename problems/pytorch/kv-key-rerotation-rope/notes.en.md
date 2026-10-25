# Solution Notes · KeyRerotation

## Problem: Eviction Breaks RoPE's "Relative Position"

The essence of RoPE is: when the key/query at two positions $p, q$ are dotted, the result depends only on the **relative distance** $q-p$ (the rotation angles subtract). KV compression picks a batch of tokens out of positions $0..L-1$ and **re-lays them out as $0..n\_kept-1$ in the new cache**. If you just gather and move on, the kept keys still carry their "original-position angles" while the query uses "new-position angles" — their difference is no longer the true relative distance, and the attention scores get misaligned.

## Solution: Add the Angle Difference Back

Suppose the key at original position $p$ has already been rotated by $R(p)$. To make it behave as if it were at **new position $i$**, multiply by an additional $R(i-p)$:

$$k' = R(\delta)\,k,\quad \delta = i - p$$

Expanded into the per-dimension rotation this is exactly the formula in the statement: `out = k*cos(emb) + rotate_half(k)*sin(emb)`, `emb = [δ·inv_freq, δ·inv_freq]`. This is isomorphic to the standard RoPE forward pass, with "position" replaced by "position difference".

## The rotate_half Direction Convention

Different repos may define `rotate_half` with an opposite sign (HF uses `[-x2, x1]`). This problem pins the definition in the statement; the oracle and the implementation both use the same one, so simply following the statement will match.

## A Negative delta Is Fine

Kept positions are always $\ge$ their new positions (since intermediate tokens were dropped), so $\delta=i-p\le 0$. $\cos$ is an even function and $\sin$ is an odd function, so a negative angle automatically equals "reverse rotation" — no special casing needed.

## Two Identity Self-checks

- **Contiguous keep** $[0,1,...,L-1]$: $\delta\equiv 0$ → the output should equal the original `keys` exactly (`check_identity_when_contiguous`).
- **Position 0 kept at new position 0**: that row has $\delta=0$ → unchanged.

These two are the fastest assertions for debugging flipped signs or concatenation order.

## Relation to ChunkPress/ChunkKV

KeyRerotation itself **does not decide which tokens to keep**; it is a "post-processing wrapper": after any ScorerPress selects indices, a re-rotation pass can be layered on to keep RoPE self-consistent. This problem extracts that layer on its own and takes the indices directly as input.
