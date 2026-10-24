# 解题说明 · ThinK

## 通道打分公式的直觉

Attention 分数是 $q^\top k$——**逐通道乘积求和**。若某个通道 $d$ 上 $q$ 与 $k$ 都近似 0，它对分数的贡献就接近 0，可以直接丢弃。ThinK 用均方（channel-wise energy）来近似 $|q_d| \cdot |k_d|$：

$$ \text{energy}_d = \underbrace{\frac{1}{W}\sum_w q_{w,d}^2}_{\text{query 侧}} \cdot \underbrace{\frac{1}{L}\sum_\ell k_{\ell,d}^2}_{\text{key 侧}} $$

乘积近似的是"这一通道在 dot 里的平均贡献量级"。

## 为什么 kv-group 要 mean

真实 GQA 里 queries 是 `num_heads`、keys 是 `num_kv_heads`。kvpress 的实现：把 queries 沿 group 维 reshape 后 mean → 得到 per-kv-head 的 channel energy。题目里 `H_q == H_kv`，跳过这一步。

## 与 Token-wise 剪枝的区别

Token-wise（KNorm / SnapKV）压 shape $(L, D)$ 的 **L 维**——保留的 token 少；channel-wise 压 **D 维**——保留的通道少。两种压缩正交，可以 `ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])` 得到 4x 内存节省。

## 为什么"置 0"而不是 gather

kvpress 官方注释：直接 scatter 0 保留 shape 便于组合、内存收益留给未来实现；题目沿用此约定。判题只看数值。
