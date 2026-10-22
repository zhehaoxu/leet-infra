import math

import torch
import torch.nn.functional as F


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    # 单 query 对两个等分 key：softmax 权重各 1/2
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    v = torch.tensor([[[3.0, 0.0], [1.0, 0.0]]])
    y = solve(q, k, v)
    expected = torch.tensor([[[2.0, 0.0]]])
    assert torch.allclose(y, expected, atol=1e-6), f"got {y.tolist()}, expected {expected.tolist()}"


def check_matches_torch(solve):
    torch.manual_seed(42)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 4)
    y = solve(q, k, v)
    exp = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(y, exp, atol=1e-5), "与 F.scaled_dot_product_attention 不一致"


def check_matches_torch_with_mask(solve):
    torch.manual_seed(1)
    q = torch.randn(2, 3, 6, 8)
    k = torch.randn(2, 3, 6, 8)
    v = torch.randn(2, 3, 6, 8)
    mask = torch.zeros(2, 3, 6, 6, dtype=torch.bool)
    mask[..., 0, 1:] = True  # 第 0 行屏蔽自己(col0)，只能看其他 key（True=keep）
    y = solve(q, k, v, mask)
    exp = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(y, exp, atol=1e-5), "带 bool mask 时不一致"


def check_scale_matters(solve):
    # sqrt(d) 缩放：去掉缩放会改变注意力分布（大 d 时尤其明显）
    torch.manual_seed(2)
    q = torch.randn(1, 4, 64) * 3
    k = torch.randn(1, 4, 64) * 3
    v = torch.randn(1, 4, 64)
    y = solve(q, k, v)
    un = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), -1) @ v
    assert not torch.allclose(y, un, atol=1e-3), "必须按 1/sqrt(d_k) 缩放"


def check_mask_keep_semantics(solve):
    # torch bool mask：True=允许注意力，False=屏蔽(置 -inf 后概率 0)
    torch.manual_seed(3)
    q = torch.randn(1, 4, 16)
    k = torch.randn(1, 4, 16)
    v = torch.eye(4)[None]  # (1, 4, 4)：第 i 行只可见自己的 value 就是 e_i
    # 只保留对角可见（True），其余屏蔽（False）
    mask = torch.eye(4, dtype=torch.bool)[None]
    y = solve(q, k, v, mask)
    # 每行只可见对角：输出必须恰等于该行对应的 v 向量
    assert torch.allclose(y[0], torch.eye(4), atol=1e-6), "被屏蔽(False)的位应概率为 0"
    # 与 torch 内置完全一致
    exp = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(y, exp, atol=1e-5), "True=keep 语义应与 torch 一致"


def check_stable_for_large_logits(solve):
    # 大内积：不减 max 的 exp 会上溢成 NaN
    q = torch.full((1, 2, 8), 100.0)
    k = torch.full((1, 2, 8), 100.0)
    v = torch.randn(1, 2, 8)
    y = solve(q, k, v)
    assert torch.isfinite(y).all(), "未做 max 平移导致 NaN"
    assert torch.allclose(y, v.mean(dim=1, keepdim=True).expand_as(y), atol=1e-5), "两 key 应等权"


check_basic.example = True
