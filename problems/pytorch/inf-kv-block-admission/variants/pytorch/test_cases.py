import random

import torch


def _oracle(num_blocks, block_size, prompt_lens, max_new, max_steps):
    """独立 oracle：不数块数，直接为每个请求分配具体块号、按页号判扩容。"""
    n = len(prompt_lens)
    free_ids = list(range(num_blocks))
    blocks = [[] for _ in range(n)]   # 每个请求持有的具体块号
    gen = [0] * n
    admit = [-1] * n
    finish = [-1] * n
    stalls = [0] * n
    peak = 0

    for step in range(max_steps):
        for r in range(n):
            if admit[r] < 0 or finish[r] >= 0 or gen[r] >= max_new[r]:
                continue
            page = (prompt_lens[r] + gen[r]) // block_size
            if page >= len(blocks[r]):
                if not free_ids:
                    stalls[r] += 1
                    continue
                blocks[r].append(free_ids.pop())
                peak = max(peak, num_blocks - len(free_ids))
            gen[r] += 1
            if gen[r] == max_new[r]:
                finish[r] = step
                free_ids.extend(blocks[r])
                blocks[r] = []
        for r in range(n):
            if admit[r] >= 0:
                continue
            need = -(-prompt_lens[r] // block_size)
            if need > len(free_ids):
                continue
            for _ in range(need):
                blocks[r].append(free_ids.pop())
            admit[r] = step
            peak = max(peak, num_blocks - len(free_ids))
            if max_new[r] == 0:
                finish[r] = step
                free_ids.extend(blocks[r])
                blocks[r] = []
        if all(f >= 0 for f in finish):
            break

    return admit, finish, stalls, peak


def check_example(solve):
    torch.manual_seed(0)
    admit, finish, stalls, peak = solve(
        3, 4, torch.tensor([4, 6]), torch.tensor([4, 2]), 10
    )
    assert admit.tolist() == [0, 0], f"两个请求都应在第 0 步准入，got {admit.tolist()}"
    assert finish.tolist() == [6, 2], f"完成步应为 [6, 2]，got {finish.tolist()}"
    assert stalls.tolist() == [2, 0], f"r0 应顺延 2 步，got {stalls.tolist()}"
    assert int(peak) == 3, f"峰值占用应为 3 块，got {int(peak)}"


check_example.example = True


def check_roomy_runs_without_stall(solve):
    """块充足时没有任何顺延，完成步恰好是准入步 + max_new。"""
    torch.manual_seed(1)
    prompts = torch.tensor([7, 1, 16, 3])
    wants = torch.tensor([3, 5, 1, 2])
    admit, finish, stalls, peak = solve(64, 4, prompts, wants, 32)
    assert bool((admit == 0).all()), f"块充足应全部在第 0 步准入：{admit.tolist()}"
    assert torch.equal(finish, wants), f"完成步应为 max_new：{finish.tolist()}"
    assert bool((stalls == 0).all()), f"不该有顺延：{stalls.tolist()}"
    # 峰值不等于「各自最坏占用之和」：先完成的请求会把块还回来
    want = _oracle(64, 4, prompts.tolist(), wants.tolist(), 32)
    assert int(peak) == want[3], f"峰值应为 {want[3]}，got {int(peak)}"
    assert int(peak) < sum((p + w + 3) // 4 for p, w in zip(prompts.tolist(), wants.tolist()))


def check_edge_cases(solve):
    """装不下的大请求被跳过；max_new=0 当场完成；步数耗尽记 -1。"""
    torch.manual_seed(2)
    admit, finish, stalls, peak = solve(
        2, 2, torch.tensor([10, 2]), torch.tensor([1, 1]), 5
    )
    assert admit.tolist() == [-1, 0], f"大请求应始终挤不进来，got {admit.tolist()}"
    assert finish.tolist() == [-1, 1], f"got {finish.tolist()}"
    assert int(peak) == 2, f"峰值应为 2，got {int(peak)}"

    admit, finish, stalls, peak = solve(
        2, 4, torch.tensor([4]), torch.tensor([0]), 1
    )
    assert admit.tolist() == [0] and finish.tolist() == [0], "max_new=0 应在准入步就完成"
    assert int(stalls[0]) == 0 and int(peak) == 1, "只占 prefill 的 1 块"

    admit, finish, stalls, peak = solve(
        2, 4, torch.tensor([4, 4]), torch.tensor([10, 10]), 3
    )
    assert admit.tolist() == [0, 0], f"got {admit.tolist()}"
    assert finish.tolist() == [-1, -1], "块一直不够，3 步内应都完不成"
    assert stalls.tolist() == [2, 2], f"两次尝试各顺延一步，got {stalls.tolist()}"
    assert int(peak) == 2, f"got {int(peak)}"


def check_random_against_oracle(solve):
    """随机负载与「具体块号分配」oracle 对拍。"""
    rng = random.Random(2026)
    for _ in range(40):
        n = rng.randint(1, 5)
        block_size = rng.randint(1, 5)
        num_blocks = rng.randint(1, 10)
        prompts = [rng.randint(0, 12) for _ in range(n)]
        wants = [rng.randint(0, 6) for _ in range(n)]
        steps = rng.randint(0, 25)
        got = solve(
            num_blocks,
            block_size,
            torch.tensor(prompts, dtype=torch.int64),
            torch.tensor(wants, dtype=torch.int64),
            steps,
        )
        want = _oracle(num_blocks, block_size, prompts, wants, steps)
        assert got[0].tolist() == want[0], f"准入步不一致：{got[0].tolist()} vs {want[0]}"
        assert got[1].tolist() == want[1], f"完成步不一致：{got[1].tolist()} vs {want[1]}"
        assert got[2].tolist() == want[2], f"顺延数不一致：{got[2].tolist()} vs {want[2]}"
        assert int(got[3]) == want[3], f"峰值不一致：{int(got[3])} vs {want[3]}"


def check_invalid_inputs(solve):
    one = torch.tensor([4])
    for bad in (
        lambda: solve(0, 4, one, one, 5),
        lambda: solve(4, 0, one, one, 5),
        lambda: solve(4, 4, one, one, -1),
        lambda: solve(4, 4, torch.tensor([[4]]), one, 5),
        lambda: solve(4, 4, one, torch.tensor([4, 4]), 5),
        lambda: solve(4, 4, torch.tensor([-1]), one, 5),
        lambda: solve(4, 4, one, torch.tensor([-2]), 5),
    ):
        try:
            bad()
            raise AssertionError("非法输入应抛 ValueError")
        except ValueError:
            pass
