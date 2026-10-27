import torch


def _ceil_div(a, b):
    return (a + b - 1) // b


def solve(num_blocks, block_size, prompt_lens, max_new, max_steps):
    if num_blocks < 1 or block_size < 1:
        raise ValueError("num_blocks / block_size 必须为正")
    if max_steps < 0:
        raise ValueError("max_steps 不能为负")
    if not isinstance(prompt_lens, torch.Tensor) or not isinstance(max_new, torch.Tensor):
        raise ValueError("prompt_lens / max_new 必须是张量")
    if prompt_lens.dim() != 1 or max_new.dim() != 1:
        raise ValueError("prompt_lens / max_new 必须是一维")
    if prompt_lens.shape != max_new.shape:
        raise ValueError("prompt_lens 与 max_new 长度必须一致")
    if prompt_lens.numel() and int(prompt_lens.min()) < 0:
        raise ValueError("prompt_lens 不能为负")
    if max_new.numel() and int(max_new.min()) < 0:
        raise ValueError("max_new 不能为负")

    n = int(prompt_lens.numel())
    prompts = prompt_lens.to(torch.int64).tolist()
    wants = max_new.to(torch.int64).tolist()

    free = num_blocks
    held = [0] * n
    gen = [0] * n
    admit = [-1] * n
    finish = [-1] * n
    stalls = [0] * n
    peak = 0

    for step in range(max_steps):
        # 解码阶段：已经跑着的请求逐个尝试吐一个 token
        for r in range(n):
            if admit[r] < 0 or finish[r] >= 0 or gen[r] >= wants[r]:
                continue
            need = _ceil_div(prompts[r] + gen[r] + 1, block_size)
            if need > held[r]:
                extra = need - held[r]
                if extra > free:
                    stalls[r] += 1
                    continue
                free -= extra
                held[r] += extra
                peak = max(peak, num_blocks - free)
            gen[r] += 1
            if gen[r] == wants[r]:
                finish[r] = step
                free += held[r]
                held[r] = 0

        # 准入阶段：按 id 升序把能装下的等待请求拉起来，装不下的跳过继续看后面的
        for r in range(n):
            if admit[r] >= 0:
                continue
            need = _ceil_div(prompts[r], block_size)
            if need > free:
                continue
            free -= need
            held[r] = need
            admit[r] = step
            peak = max(peak, num_blocks - free)
            if wants[r] == 0:
                finish[r] = step
                free += need
                held[r] = 0

        if all(f >= 0 for f in finish):
            break

    return (
        torch.tensor(admit, dtype=torch.int64),
        torch.tensor(finish, dtype=torch.int64),
        torch.tensor(stalls, dtype=torch.int64),
        torch.tensor(peak, dtype=torch.int64),
    )
