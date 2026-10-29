import torch


def solve(x, gate_w, bias, group_count, top_groups, k, gamma, update_bias=False):
    """中文契约 / English contract
    x [T,D], gate_w [E,D], bias [E]: CPU float64；E 必须被 group_count 整除且组内至少 2 个专家 /
        CPU float64; E must divide by group_count and every group needs at least 2 experts.
    group_count/top_groups/k：真正的 Python int，k 在 [1, top_groups*group_size]；gamma >= 0 /
        true Python ints, k in [1, top_groups*group_size]; gamma >= 0.
    结构、k 或 gamma 非法时抛 ValueError；不修改输入，接受非连续张量 /
        raise ValueError for invalid structure, k, or gamma; mutate no inputs and accept noncontiguous tensors.
    sigmoid 分数 + bias 只参与选择（分组 top-2 和后组内 top-k），权重只由原始 sigmoid 分数归一化 /
        sigmoid scores plus bias used only for selection (group top-2 sum, then in-group top-k);
        weights normalize raw sigmoid scores only.
    返回 / Returns (weights [T,k], indices [T,k], scores [T,E], group_scores [T,G],
        selected_groups [T,top_groups], load [E], new_bias [E])；update_bias=False 时返回独立副本 /
        new_bias is an independent copy when update_bias is False.
    """
    raise NotImplementedError
