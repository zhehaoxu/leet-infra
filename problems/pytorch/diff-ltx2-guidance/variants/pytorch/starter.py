def solve(
    pred_cond,
    pred_uncond,
    pred_stg,
    pred_modality,
    step,
    num_inference_steps,
    cfg_scale=1.0,
    stg_scale=0.0,
    modality_scale=1.0,
    start=0.0,
    stop=1.0,
):
    """LTX-2 三路增量引导的单模态合成。

    四段同形预测（第 0 维为 batch）：
      pred_cond       —— 条件预测（基准）；
      pred_uncond     —— 无条件预测，配合 cfg_scale；
      pred_stg        —— 扰动若干 transformer block 后的条件预测，配合 stg_scale；
      pred_modality   —— 关掉 A2V/V2A 跨模态注意力后的条件预测，配合 modality_scale。

    三个尺度共用同一个左闭右开窗口：
      within = int(start*N) <= step < int(stop*N)，N = num_inference_steps
    各分量是否生效：
      cfg_on = within and not isclose(cfg_scale, 1.0)
      stg_on = within and not isclose(stg_scale, 0.0)
      mod_on = within and not isclose(modality_scale, 1.0)

    合成采用增量（delta）形式，三项都以 pred_cond 为基准、彼此可加：
      pred = pred_cond
             + (cfg_scale - 1)      * (pred_cond - pred_uncond)      # cfg_on 时
             +  stg_scale           * (pred_cond - pred_stg)         # stg_on 时
             + (modality_scale - 1) * (pred_cond - pred_modality)    # mod_on 时
    注意 cfg 与 modality 是「以 1 为恒等」的尺度（差的是 1），stg 是「以 0 为恒等」的尺度。

    某个分量关闭时对应输入可以传 None；若分量生效却传 None，抛 ValueError。
    返回 float32 张量，形状与 pred_cond 相同。
    """
    # TODO
    raise NotImplementedError
