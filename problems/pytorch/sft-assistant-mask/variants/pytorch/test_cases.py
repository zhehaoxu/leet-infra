import torch


def _oracle(token_ids, role_ids, assistant_id, eos_id, supervise_eos):
    B, L = role_ids.shape
    out = []
    for b in range(B):
        row = []
        for i in range(L):
            is_asst = int(role_ids[b, i]) == assistant_id
            run_last = (i == L - 1) or int(role_ids[b, i + 1]) != assistant_id
            drop_eos = (not supervise_eos) and run_last and int(token_ids[b, i]) == eos_id
            row.append(1.0 if (is_asst and not drop_eos) else 0.0)
        out.append(row)
    return torch.tensor(out, dtype=torch.float32)


TOK = torch.tensor([[5, 6, 7, 8, 99, 0], [1, 2, 99, 3, 4, 0]])
ROLE = torch.tensor([[0, 1, 2, 2, 2, 3], [1, 2, 2, 2, 3, 3]])


def check_example(solve):
    out = solve(TOK, ROLE, 2, 99, False)
    expect = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(out, expect), out.tolist()


check_example.example = True


def check_supervise_eos_true(solve):
    out = solve(TOK, ROLE, 2, 99, True)
    expect = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    assert torch.allclose(out, expect), out.tolist()


def check_eos_in_middle_kept(solve):
    # assistant run 中间出现 eos(99)，仅段尾才可能被丢
    tok = torch.tensor([[1, 2, 99, 3, 4]])
    role = torch.tensor([[2, 2, 2, 2, 2]])
    out = solve(tok, role, 2, 99, False)
    # 段尾 token=4 != 99 -> 整段保留
    assert torch.allclose(out, torch.ones(1, 5)), out.tolist()


def check_no_assistant_all_zero(solve):
    tok = torch.tensor([[7, 8, 9]])
    role = torch.tensor([[0, 1, 3]])
    out = solve(tok, role, 2, 99, True)
    assert torch.allclose(out, torch.zeros(1, 3)), out.tolist()


def check_matches_oracle(solve):
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        role = torch.randint(0, 4, (3, 8), generator=g)
        tok = torch.randint(0, 120, (3, 8), generator=g)
        for sp in (True, False):
            y = solve(tok, role, 2, 99, sp)
            assert torch.allclose(y, _oracle(tok, role, 2, 99, sp)), (seed, sp)
