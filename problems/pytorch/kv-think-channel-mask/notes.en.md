# Solution Notes · ThinK

## Intuition behind the channel scoring formula

The attention score is $q^\top k$—a **sum of per-channel products**. If both $q$ and $k$ are close to 0 on some channel $d$, that channel's contribution to the score is close to 0 and it can be dropped outright. ThinK approximates $|q_d| \cdot |k_d|$ with mean squares (channel-wise energy):

$$ \text{energy}_d = \underbrace{\frac{1}{W}\sum_w q_{w,d}^2}_{\text{query side}} \cdot \underbrace{\frac{1}{L}\sum_\ell k_{\ell,d}^2}_{\text{key side}} $$

The product approximates "the magnitude of this channel's average contribution inside the dot product".

## Why kv-group needs a mean

In real GQA, queries have `num_heads` heads and keys have `num_kv_heads` heads. kvpress's implementation reshapes the queries along the group dimension and takes the mean → per-kv-head channel energy. Here `H_q == H_kv`, so that step is skipped.

## Difference from token-wise pruning

Token-wise methods (KNorm / SnapKV) compress the **L dimension** of a $(L, D)$ shape—fewer tokens are kept; channel-wise methods compress the **D dimension**—fewer channels are kept. The two kinds of compression are orthogonal, so `ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])` yields 4x memory savings.

## Why "set to 0" instead of gather

The official kvpress comment: scattering 0 in place keeps the shape, which is convenient for composition, and the memory savings are left to future implementations; this problem follows the same convention. Grading only looks at the values.
