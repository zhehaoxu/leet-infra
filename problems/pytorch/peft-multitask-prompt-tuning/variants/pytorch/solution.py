import torch


def solve(shared_prompt, task_prompts, task_ids):
    """MultiTask Prompt Tuning 拼接：P_b = [shared_prompt; task_prompts[task_ids[b]]]。
    shared_prompt:(s,H) task_prompts:(T,t,H) task_ids:(B,)。返回 P:(B,s+t,H)。"""
    selected = task_prompts[task_ids]                       # (B, t, H)
    shared = shared_prompt.unsqueeze(0).expand(selected.size(0), -1, -1)  # (B, s, H)
    return torch.cat([shared, selected], dim=1)             # (B, s+t, H)
