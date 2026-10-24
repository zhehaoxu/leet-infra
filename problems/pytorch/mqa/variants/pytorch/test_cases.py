import torch
import torch.nn.functional as F


def _sdpa_ref(q, k, v, num_heads):
    b, lq, hid = q.shape
    dh = hid // num_heads
    lk = k.shape[1]
    qh = q.view(b, lq, num_heads, dh).transpose(1, 2)
    kh = k.view(b, lk, 1, dh).transpose(1, 2)
    vh = v.view(b, lk, 1, dh).transpose(1, 2)
    r = F.scaled_dot_product_attention(qh, kh, vh, enable_gqa=True)
    return r.transpose(1, 2).reshape(b, lq, hid)


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_matches_torch(solve):
    torch.manual_seed(0)
    b, lq, lk, nh, dh = 2, 5, 7, 4, 3
    hid = nh * dh
    for _ in range(3):
        q = torch.randn(b, lq, hid)
        k = torch.randn(b, lk, dh)   # 单个共享 KV 头
        v = torch.randn(b, lk, dh)
        got = solve(q, k, v, nh)
        assert torch.allclose(got, _sdpa_ref(q, k, v, nh), atol=1e-5)


def check_edge_example(solve):
    q = torch.tensor([[[1.0, 0.0, 2.0, 1.0]]])
    k = torch.tensor([[[0.5, 1.0], [1.0, 0.0]]])
    v = torch.tensor([[[1.0, 2.0], [0.0, 1.0]]])
    got = solve(q, k, v, 2).flatten()
    ref = torch.tensor([0.412521, 1.412521, 0.5, 1.5])
    assert torch.allclose(got, ref, atol=1e-5), got.tolist()


def check_output_shape(solve):
    torch.manual_seed(1)
    nh, dh = 3, 4
    q = torch.randn(2, 6, nh * dh)
    k = torch.randn(2, 9, dh)
    v = torch.randn(2, 9, dh)
    assert solve(q, k, v, nh).shape == (2, 6, nh * dh)


def check_single_head_equals_attention(solve):
    # num_heads=1 时 MQA 退化为普通单头注意力
    torch.manual_seed(2)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 6, 8)
    v = torch.randn(1, 6, 8)
    got = solve(q, k, v, 1)
    s = q @ k.transpose(-2, -1) / (8 ** 0.5)
    ref = torch.softmax(s, dim=-1) @ v
    assert torch.allclose(got, ref, atol=1e-5)


check_matches_torch.example = True
