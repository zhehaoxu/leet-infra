import torch


def solve(cls_attn):
    """LLaVA-PruMerge 的 IQR 自适应 token 数 / IQR-adaptive kept-token count.

    中文契约：
    - cls_attn：CPU float32/float64 的 [B,N]，是 CLS→patch 的注意力，已由调用方
      在 head 维平均完毕（上游为 softmax 概率，但本题只要求有限浮点值）。
      B=1..4，N=1..24。
    - 步骤一（自适应比率，逐行转写上游 `outlier_dectection`）：
      1. 把**整个 [B,N]** 摊平成一条序列，并先 cast 成 **float32**（float64 输入
         也在这一步被降精度，这是上游行为点）。
      2. Q1 = 该序列的第 25 百分位、Q3 = 第 75 百分位，均用 numpy 的默认
         **linear 插值**口径（q/100*(n-1) 虚位线性内插）。
      3. IQR = Q3 - Q1，上界 upper = Q3 + 1.5 * IQR（Tukey 1.5 倍，只看上尾；
         被注释掉的下界不出题）。
      4. ratio = 序列中**严格大于** upper 的元素个数 / 序列总长，是一个
         跨 batch 共享的 Python float。比较发生在 float32 cast 之后。
    - 步骤二（选择，与 order 536 同口径）：保留个数 k = int(N * ratio)，
      **向下截断**。与 536 不同，k == 0 在这里是**合法结果**（上游自适应路径
      允许一个 token 都不保留）：此时 kept 是 [B,0] 空张量，pruned 是 0..N-1。
    - kept：每行 cls_attn 最大的 k 个下标，并列按「值降序、下标升序」判定，
      最终**按升序下标**发出；pruned = 其余下标，同样升序。两组互不重叠，
      并起来恰为 0..N-1。
    - 数学保证（测试会断言）：序列中至少有一个元素不严格大于 upper，
      因此 ratio < 1 恒成立、k <= N-1、pruned 永不为空。
    - ValueError：cls_attn 不是张量、dtype 不是 float32/float64、不是二维、
      B<1、N<1、含非有限值。
    - 接受非连续输入，不修改任何输入。
    - 返回三元组，顺序固定：
      ratio：Python float，跨 batch 共享的自适应比率；
      kept_indexs：int64 [B,k]，升序；k 可为 0；
      pruned_indexs：int64 [B,N-k]，升序，永不为空。
    - 禁止使用 numpy.percentile 之外的 scipy 依赖；np.percentile 与
      torch.argsort(stable=True) 均可用。

    English contract:
    - cls_attn is a CPU float32/float64 tensor of shape [B,N]: the CLS-to-patch
      attention already averaged over heads by the caller (upstream feeds
      softmax probabilities, but this task only requires finite floats).
      Bounds: B=1..4, N=1..24.
    - Step one (adaptive ratio, a faithful transcription of the upstream
      `outlier_dectection`): (1) flatten the WHOLE [B,N] into one sequence and
      cast it to float32 FIRST — float64 inputs lose precision here, an
      upstream behaviour point; (2) Q1/Q3 are the 25th/75th percentiles under
      numpy's default LINEAR interpolation; (3) IQR = Q3 - Q1 and
      upper = Q3 + 1.5 * IQR (Tukey 1.5, upper tail only — the commented-out
      lower bound is out of scope); (4) ratio = (# elements STRICTLY greater
      than upper) / (sequence length), one Python float shared across the
      batch. The comparison happens AFTER the float32 cast.
    - Step two (selection, same conventions as order 536): k = int(N * ratio),
      truncating. Unlike 536, k == 0 is a LEGAL result here (the adaptive path
      may keep nothing): kept is then an empty [B,0] tensor and pruned is
      0..N-1.
    - kept holds the k largest cls_attn indices per row, ties resolved as
      "value descending, index ascending", EMITTED in ascending index order;
      pruned is the ascending complement. The two sets are disjoint and union
      to 0..N-1.
    - Mathematical guarantee (asserted by the tests): at least one element is
      not strictly greater than upper, so ratio < 1 always, k <= N-1, and
      pruned is never empty.
    - Raise ValueError when cls_attn is not a tensor, its dtype is not
      float32/float64, it is not 2-D, B < 1 or N < 1, or it contains
      non-finite values.
    - Accept noncontiguous inputs and never mutate any input.
    - Return a triple in this fixed order: ratio (Python float, shared across
      the batch), kept_indexs (int64 [B,k], ascending, k may be 0), and
      pruned_indexs (int64 [B,N-k], ascending, never empty).
    - No scipy beyond numpy.percentile; np.percentile and
      torch.argsort(stable=True) are both allowed.
    """
    raise NotImplementedError
