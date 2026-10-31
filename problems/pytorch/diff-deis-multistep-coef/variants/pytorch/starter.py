import torch


def solve(
    sample,
    model_output,
    sigmas,
    step_index,
    converted_history,
    solver_order=2,
    lower_order_final=True,
    prediction_type="epsilon",
    use_flow_sigmas=False,
):
    """DEIS 多步单步更新（见题目描述）。"""
    raise NotImplementedError
