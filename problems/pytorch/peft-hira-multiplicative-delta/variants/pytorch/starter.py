def solve(weight, hira_A, hira_B, x, fan_in_fan_out=False):
    """HiRA（huggingface/peft tuners/hira）乘性高秩适配的 CPU 参考口径。

    返回 dict：delta_weight、effective_weight、hira_result、merged_weight，
    全部按输入的存储布局返回（fan_in_fan_out=True 时为 [in, out]）。
    """
    raise NotImplementedError
