import torch


def _oracle(input_ids, supervised_mask, ignore_index=-100):
    ids = input_ids.tolist()
    mk = [[1 if float(v) > 0 else 0 for v in row] for row in supervised_mask.tolist()]
    out = []
    for row, mrow in zip(ids, mk):
        L = len(row)
        shifted = row[1:] + [ignore_index]          # 先整体左移，末位补 ignore
        sel_m = mrow[1:] + [0]                       # 后继位掩码，末位视作不监督
        lab = []
        for k in range(L):
            if k < L - 1 and sel_m[k] == 1:
                lab.append(shifted[k])
            else:
                lab.append(ignore_index)
        out.append(lab)
    return torch.tensor(out, dtype=torch.long)


def check_example(solve):
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.float32)
    out = solve(ids, mask)
    assert out.tolist() == [[-100, 12, 13, -100, -100]], out.tolist()


check_example.example = True


def check_all_zero_mask(solve):
    ids = torch.tensor([[3, 4, 5, 6]])
    mask = torch.zeros(1, 4)
    out = solve(ids, mask)
    assert out.tolist() == [[-100, -100, -100, -100]], out.tolist()


def check_all_one_shift(solve):
    ids = torch.tensor([[7, 8, 9]])
    mask = torch.ones(1, 3)
    out = solve(ids, mask)
    # labels[i]=ids[i+1]: [8,9,-100]
    assert out.tolist() == [[8, 9, -100]], out.tolist()


def check_no_cross_row(solve):
    ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)
    out = solve(ids, mask)
    # 每行末列恒 -100：行0 末位不能取行1 首 token(3)，行1 末位不跨下一行
    assert out.tolist() == [[2, -100], [4, -100]], out.tolist()


def check_matches_oracle(solve):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 50, (4, 7), generator=g)
    mask = (torch.rand(4, 7, generator=g) > 0.5).float()
    for ig in (-100, -1):
        y = solve(ids, mask, ig)
        assert torch.equal(y, _oracle(ids, mask, ig)), ig
