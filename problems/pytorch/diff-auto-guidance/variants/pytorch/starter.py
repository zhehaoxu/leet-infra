def solve(
    pred_cond,
    pred_uncond,
    step,
    num_inference_steps,
    guidance_scale=1.0,
    start=0.0,
    stop=1.0,
    use_original_formulation=False,
    guidance_rescale=0.0,
):
    """AutoGuidance：锚点与失效阈值随表述方式一起翻转的引导算子。

    两个同形预测（第 0 维为 batch）：
      pred_cond   —— 完整模型的条件预测；
      pred_uncond —— 「变坏的模型」预测（AutoGuidance 里是把若干 transformer block
                     按 dropout 概率丢弃后得到的弱模型；本题把它当作显式输入）。
                     引导分量关闭时可以传 None；生效却传 None 抛 ValueError。

    生效判定（两条件同时成立才算「开」）：
      within = int(start * N) <= step < int(stop * N)，N = num_inference_steps
      close  = math.isclose(guidance_scale, 0.0)  当 use_original_formulation
               math.isclose(guidance_scale, 1.0)  否则
      on = within and not close
    注意两个阈值是跟着表述一起换的：默认表述以 1 为恒等，原论文表述以 0 为恒等。
    math.isclose 用默认 rel_tol=1e-9、abs_tol=0：与 1.0 的比较是相对的（1+1e-10 算恒等），
    与 0.0 的比较是绝对的（只有恰好 0.0 才算恒等）。

    关闭时：pred = pred_cond，且 pred_uncond 完全不参与计算与校验。
    开启时：
      shift = pred_cond - pred_uncond
      pred  = (pred_cond if use_original_formulation else pred_uncond) + guidance_scale * shift
    即默认表述把锚点放在 pred_uncond 上外推，原论文表述把锚点放在 pred_cond 上外推。

    guidance_rescale = r ∈ [0, 1]：只要 r > 0 就执行（关闭分支也执行），逐样本
    （除第 0 维外的全部维度，torch.std 默认的无偏标准差）：
      pred = r * (pred * (std(pred_cond) / std(pred))) + (1 - r) * pred

    入参约束：
      step / num_inference_steps 为 Python int（bool 非法），N > 0 且 0 <= step < N；
      start ∈ [0, 1)（严格小于 1）、stop ∈ [start, 1]；
      guidance_scale / guidance_rescale 为有限实数；use_original_formulation 为 bool；
      两张预测都必须是全有限值的 torch.Tensor、至少两维、除 batch 维外元素个数 >= 2 且形状一致。
    违反任一约束抛 ValueError。

    全部中间计算在 float64 下进行，返回 float32 张量，形状与 pred_cond 相同。
    禁止 import diffusers。
    """
    # TODO
    raise NotImplementedError
