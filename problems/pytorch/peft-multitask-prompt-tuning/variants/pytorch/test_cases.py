import torch

SHARED = torch.tensor([[1.0, 2.0], [3.0, 4.0]])                  # (2,2)
TASK = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]], [[9.0, 10.0]]])  # (3,1,2)
TASK_IDS = torch.tensor([0, 2, 1, 0], dtype=torch.int64)         # (4,)


def _oracle(shared_prompt, task_prompts, task_ids):
    """独立重算：逐样本拼接。"""
    B = task_ids.numel()
    s, H = shared_prompt.shape
    T, t, _ = task_prompts.shape
    out = torch.empty(B, s + t, H, dtype=shared_prompt.dtype)
    for b in range(B):
        tid = int(task_ids[b])
        out[b, :s] = shared_prompt
        out[b, s:] = task_prompts[tid]
    return out


def check_example(solve):
    P = solve(SHARED, TASK, TASK_IDS)
    exp = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                        [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]],
                        [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]],
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    assert torch.allclose(P, exp, atol=1e-5, rtol=1e-5), f"got {P.tolist()}"


check_example.example = True


def check_matches_oracle(solve):
    torch.manual_seed(31)
    shared = torch.randn(3, 5)
    task = torch.randn(4, 2, 5)
    ids = torch.tensor([3, 0, 2, 2, 1], dtype=torch.int64)
    P = solve(shared, task, ids)
    assert torch.allclose(P, _oracle(shared, task, ids),
                          atol=1e-5, rtol=1e-5), "与逐样本独立重算不符"


def check_single_task_broadcast(solve):
    """所有样本 task_id 相同，共享提示仍要复制。"""
    ids = torch.zeros(3, dtype=torch.int64)
    P = solve(SHARED, TASK, ids)
    assert P.shape == (3, 3, 2), "batch 维度应随 task_ids"
    for b in range(3):
        assert torch.allclose(P[b], _oracle(SHARED, TASK, ids)[b],
                              atol=1e-6, rtol=1e-6)


def check_shared_first_task_second(solve):
    """固定反例：若把顺序颠倒，结果应不同。"""
    P = solve(SHARED, TASK, TASK_IDS)
    wrong = torch.cat([TASK[TASK_IDS], SHARED.unsqueeze(0).expand(4, -1, -1)], dim=1)
    assert not torch.allclose(P, wrong, atol=1e-4, rtol=1e-4), "顺序必须是共享在前、任务在后"


def check_batch_size_one(solve):
    ids = torch.tensor([2], dtype=torch.int64)
    P = solve(SHARED, TASK, ids)
    assert P.shape == (1, 3, 2), "B=1 时形状应正确"
