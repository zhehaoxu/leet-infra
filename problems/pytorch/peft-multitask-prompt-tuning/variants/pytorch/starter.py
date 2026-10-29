import torch


def solve(shared_prompt, task_prompts, task_ids):
    """MultiTask Prompt Tuning 拼接：P_b = [shared_prompt; task_prompts[task_ids[b]]]。
    shared_prompt:(s,H) task_prompts:(T,t,H) task_ids:(B,)。返回 P:(B,s+t,H)。"""
    # TODO
    raise NotImplementedError
