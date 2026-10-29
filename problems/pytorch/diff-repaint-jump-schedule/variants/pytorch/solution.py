import torch


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须是 Python 整数")
    if value < 1:
        raise ValueError(f"{name} 必须 >= 1")
    return value


def solve(num_train_timesteps, num_inference_steps, jump_length=10, jump_n_sample=10):
    num_train = _positive_int("num_train_timesteps", num_train_timesteps)
    num_inference = min(num_train, _positive_int("num_inference_steps", num_inference_steps))
    jump_length = _positive_int("jump_length", jump_length)
    jump_n_sample = _positive_int("jump_n_sample", jump_n_sample)

    jumps = {j: jump_n_sample - 1 for j in range(0, num_inference - jump_length, jump_length)}
    timesteps = []
    t = num_inference
    while t >= 1:
        t -= 1
        timesteps.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                timesteps.append(t)

    step_ratio = num_train // num_inference
    return torch.tensor([x * step_ratio for x in timesteps], dtype=torch.int64)
