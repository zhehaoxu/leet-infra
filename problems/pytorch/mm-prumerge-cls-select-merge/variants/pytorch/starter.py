import torch


def solve(image_features, cls_attn, reduction_ratio, eps=1e-12):
    """LLaVA-PruMerge 的 CLS 选 token 与余弦残差合并 / CLS selection plus cosine residual merge.

    中文契约：
    - image_features：CPU float32/float64 [B,N,C]，仅 patch 特征，**不含 CLS 行**。
      cls_attn：与 image_features **完全相同 dtype** 的 [B,N]，是 CLS→patch 的注意力，
      已由调用方在 head 维平均完毕。B=1..3，N=1..24，C=1..12。
    - reduction_ratio：Python 实数（int/float，拒绝 bool），范围 (0,1]。
    - eps：Python 实数（拒绝 bool），必须 >0，默认 1e-12，是 L2 归一化的分母下限，
      对应 F.normalize 的 eps。其余输入保证合法。
    - 每行独立：保留个数 k = int(N * reduction_ratio)，**向下截断，不是四舍五入**；
      k == 0 时抛 ValueError。kept = cls_attn 最大的 k 个下标，并列按「值降序、
      下标升序」判定，最终**按升序下标**发出；pruned = 其余下标，同样升序。
    - 归一化：norm_j = max(||image_features[b,j]||_2, eps)，u_j = 特征 / norm_j，
      kept 与 pruned 都用同一个 eps 下限，零向量的余弦因此是 0 而不是 NaN。
    - 分配：sim[i,j] = <u_{pruned i}, u_{kept j}>；cluster[i] = argmax_j sim[i,j]，
      并列取**第一个**（最小 j）。
    - 权重：p[i,:] = sim[i,:] 沿 **整个 kept 集合** 的稳定 softmax，
      weight[i] = p[i, cluster[i]] * C ** -0.5。缩放发生在 softmax **之后**，
      并且用的是特征维 C，不是 token 数；同一 cluster 内的权重不再归一化，
      因此一个 cluster 的权重和通常不等于 1。
    - 合并：merged[j] = image_features[b, kept[j]] +
      Σ_{i: cluster[i]==j} weight[i] * image_features[b, pruned[i]]；
      没有任何 pruned 分配到的 kept token 原样返回。reduction_ratio=1 时
      pruned 为空，merged 就是升序排列的原始 patch。
    - ValueError：reduction_ratio 非实数或落在 (0,1] 之外、int(N*reduction_ratio)==0、
      N<1、C<1、cls_attn 与 image_features 形状不匹配、cls_attn 不是 [B,N]、
      image_features 不是三维、eps 非实数或 <=0、dtype 不是 float32/float64
      或两者不一致、含非有限值、B<1。
    - 接受非连续输入，不修改任何输入，不强制连续化。
    - 返回六元组，顺序固定：
      merged：[B,k,C]，与输入同 dtype/device；
      kept_indexs：int64 [B,k]，每行保留的原始 patch 下标，升序；
      pruned_indexs：int64 [B,N-k]，每行被合并掉的原始下标，升序；
      cluster：int64 [B,N-k]，每个 pruned 落在 kept_indexs 第几列；
      weights：与输入同浮点 dtype [B,N-k]，已含 C ** -0.5；
      scores：与输入同浮点 dtype [B,N-k]，scores[b,i] = sim[b,i,cluster[b,i]]，
      即最大余弦相似度，用作判定台账。
    禁止 torch.nn.functional.scaled_dot_product_attention；
    F.softmax、F.normalize、torch.norm 与 scatter_add 均可用。

    English contract:
    - image_features is a CPU float32/float64 tensor [B,N,C] holding patch
      features only, with NO CLS row. cls_attn has the very same dtype and shape
      [B,N]: the CLS-to-patch attention, already averaged over heads by the
      caller. Bounds: B=1..3, N=1..24, C=1..12.
    - reduction_ratio is a Python real (int or float, bool excluded) in (0,1].
      eps is a Python real (bool excluded) greater than 0, default 1e-12, the
      floor of the L2 denominator exactly like F.normalize's eps. Other inputs
      are valid.
    - Work row by row: k = int(N * reduction_ratio), which TRUNCATES and does not
      round; raise ValueError when k evaluates to 0. kept is the set of the k
      largest cls_attn entries, ties resolved as "value descending, then index
      ascending", EMITTED in ascending index order; pruned is the complement,
      also ascending.
    - Normalization: norm_j = max(||image_features[b,j]||_2, eps) and
      u_j = feature / norm_j for kept and pruned alike, so a zero vector yields a
      cosine of 0 instead of NaN.
    - Assignment: sim[i,j] = <u_{pruned i}, u_{kept j}> and
      cluster[i] = argmax_j sim[i,j], taking the FIRST index on ties.
    - Weights: p[i,:] is the stable softmax of sim[i,:] over the WHOLE kept set,
      and weight[i] = p[i, cluster[i]] * C ** -0.5. The scaling happens AFTER the
      softmax and uses the feature dimension C, not the token count. Weights are
      not renormalized inside a cluster, so a cluster's weights usually do not
      sum to 1.
    - Merge: merged[j] = image_features[b, kept[j]] +
      sum over i with cluster[i]==j of weight[i] * image_features[b, pruned[i]].
      A kept token with no assigned pruned token is returned verbatim. With
      reduction_ratio=1 the pruned set is empty and merged is just the ascending
      original patches.
    - Raise ValueError when reduction_ratio is not real or outside (0,1], when
      int(N * reduction_ratio) == 0, when N < 1 or C < 1 or B < 1, when cls_attn
      does not match image_features' shape, when image_features is not 3-D or
      cls_attn is not [B,N], when eps is not real or <= 0, when the dtypes are
      not float32/float64 or differ from each other, or when any input value is
      non-finite.
    - Accept noncontiguous inputs and never mutate any input.
    - Return a six-tuple in this exact order:
      merged: [B,k,C], input dtype/device;
      kept_indexs: int64 [B,k], the surviving original patch indices, ascending;
      pruned_indexs: int64 [B,N-k], the merged-away original indices, ascending;
      cluster: int64 [B,N-k], the kept column each pruned token falls into;
      weights: same floating dtype [B,N-k], already scaled by C ** -0.5;
      scores: same floating dtype [B,N-k], scores[b,i] = sim[b,i,cluster[b,i]],
      i.e. the maximum cosine similarity, kept as a judging ledger.
    torch.nn.functional.scaled_dot_product_attention is forbidden;
    F.softmax, F.normalize, torch.norm and scatter_add are allowed.
    """
    raise NotImplementedError
