import torch


def solve(hidden_states, attention, image_start, image_len, r):
    """FastV 第 K 层视觉 token 剪枝 / FastV visual token pruning at layer K.

    中文契约：
    - hidden_states：CPU float32/float64 [B,L,D]，第 K 层输入。
      attention：与 hidden_states **完全相同 dtype** 的 float32/float64 [B,H,L,L]，
      是第 K 层已经 softmax 的注意力矩阵，元素 [b,h,i,j] 为 token i 对 token j 的概率。
      B=1..3，L=2..16，D=1..4，H=1..4。
    - image_start：非负 Python int（视觉段起始下标，含 system+question 前缀长度）；
      image_len：正 Python int，且 image_start+image_len<=L。两者都拒绝 bool。
    - r：Python 实数（int/float，拒绝 bool），表示**丢弃比例**，范围 [0,1)。
      保留比例是 1-r。其余输入保证合法。
    - 每行独立打分：score[b,i] = attention[b,:,-1,image_start+i] 在 head 维的平均，
      即只取**最后一个 query 行**的列元素，先对 head 求均值，i=0..image_len-1。
      保留个数 k = round(image_len*(1-r))，使用 Python round 的四舍六入五取偶语义，
      再夹到 [0,image_len]；k 可以为 0（全部视觉 token 被剪）。
    - 只有视觉段内的 token 参与竞争：image_start 之前和 image_start+image_len
      之后的文本永不剪除。并列分数按**分数降序、下标升序**取前 k 个，
      选中的视觉下标必须**再按升序**输出（保持因果顺序），不是分数顺序。
    - 幸存 token 保留**原始位置编号**（跳号保留，不能从 0 重新编号）。
    - ValueError：hidden_states/attention 的 dtype 不是 float32/float64 或两者不一致、
      含非有限值、形状不合法（attention 不是 [B,H,L,L] 且后两维等于 L、H<1）、
      r 非实数或落在 [0,1) 之外、image_start<0、image_len<1、
      image_start+image_len>L。
    - 接受非连续输入，不修改任何输入，不强制连续化。
    - 返回四元组，顺序固定：
      pruned_hidden_states：[B,L',D]，与输入同 dtype/device，L'=L-(image_len-k)；
      keep_indexs：int64 [B,L']，每行幸存 token 的原始下标，升序；
      position_ids：int64 [B,L']，等于该行的 keep_indexs（跳号保留）；
      score：与输入同浮点 dtype [B,image_len]，每行视觉段的 head 平均末行分数。
    禁止 torch.nn.functional.scaled_dot_product_attention；
    torch.sort/argsort/index_select/gather 与 F.softmax 均可用。

    English contract:
    - hidden_states is a CPU float32/float64 tensor [B,L,D], the layer-K input.
      attention has the very same dtype [B,H,L,L]: the layer-K post-softmax
      attention matrix, entry [b,h,i,j] being the probability of i attending to
      j. Bounds: B=1..3, L=2..16, D=1..4, H=1..4.
    - image_start is a non-negative Python int (the visual span offset, i.e. the
      prefix length); image_len is a positive Python int with
      image_start+image_len<=L. bool is rejected for both.
    - r is a Python real (int or float, bool excluded) in [0,1): the DROP
      fraction, so 1-r is the keep fraction. Other inputs are valid.
    - Score each row independently: score[b,i] is the mean over heads of the
      LAST query row's column image_start+i, i.e.
      attention[b,:,-1,image_start+i].mean(0), i=0..image_len-1.
      The keep count is k=round(image_len*(1-r)) with Python round's
      half-to-even behaviour, clamped into [0,image_len]; k may be 0, which
      prunes the whole visual span.
    - Only tokens inside the visual span compete. Text before image_start and
      after image_start+image_len is never pruned. Select the top k scores with
      ties broken by (score descending, index ascending), then RE-EMIT the kept
      visual indices in ascending index order, not in score order.
    - Survivors keep their ORIGINAL indices as position ids: the gaps stay.
      Do not renumber from 0.
    - Raise ValueError when hidden_states/attention are not float32/float64 or
      their dtypes differ, when either contains a non-finite value, when shapes
      are illegal (attention is not [B,H,L,L] with the last two dims equal to L,
      or H<1), when r is not real or outside [0,1), when image_start<0,
      image_len<1, or image_start+image_len>L.
    - Accept noncontiguous inputs and never mutate any input.
    - Return a four-tuple in this exact order:
      pruned_hidden_states: [B,L',D], input dtype/device, L'=L-(image_len-k);
      keep_indexs: int64 [B,L'], the surviving original indices, ascending;
      position_ids: int64 [B,L'], equal to that row of keep_indexs (gaps kept);
      score: same floating dtype [B,image_len], the head-mean last-row scores.
    torch.nn.functional.scaled_dot_product_attention is forbidden;
    torch.sort/argsort/index_select/gather and F.softmax are allowed.
    """
    raise NotImplementedError
