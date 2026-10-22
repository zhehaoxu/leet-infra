import torch


# 样例用例：点「运行」只执行标了 example = True 的用例；「提交」执行全部用例
def check_basic(solve):
    cache_k = torch.tensor([[[1.0], [2.0]]])   # (1, L=2, 1)
    cache_v = torch.tensor([[[10.0], [20.0]]])
    new_k = torch.tensor([[[3.0]]])
    new_v = torch.tensor([[[30.0]]])
    k, v = solve(cache_k, cache_v, new_k, new_v)
    assert torch.equal(k, torch.tensor([[[1.0], [2.0], [3.0]]])), f"got {k.tolist()}"
    assert torch.equal(v, torch.tensor([[[10.0], [20.0], [30.0]]])), f"got {v.tolist()}"


def check_matches_cat(solve):
    torch.manual_seed(42)
    cache_k = torch.randn(2, 3, 4, 8)
    cache_v = torch.randn(2, 3, 4, 8)
    new_k = torch.randn(2, 3, 1, 8)
    new_v = torch.randn(2, 3, 1, 8)
    k, v = solve(cache_k, cache_v, new_k, new_v)
    assert torch.equal(k, torch.cat([cache_k, new_k], dim=-2)), "K 应沿序列维拼接"
    assert torch.equal(v, torch.cat([cache_v, new_v], dim=-2)), "V 应沿序列维拼接"


def check_empty_cache(solve):
    new_k = torch.randn(1, 1, 16)
    new_v = torch.randn(1, 1, 16)
    k, v = solve(None, None, new_k, new_v)
    assert torch.equal(k, new_k) and torch.equal(v, new_v), "空缓存应直接返回新值"
    empty = new_k.new_zeros(1, 0, 16)
    k2, v2 = solve(empty, empty, new_k, new_v)
    assert k2.shape == (1, 1, 16) and torch.equal(k2, new_k), "0 长缓存等价于空缓存"


def check_incremental_decoding(solve):
    # 模拟逐 token 解码：连续 5 步追加，缓存长度与内容都对
    torch.manual_seed(0)
    ck = cv = None
    for t in range(5):
        nk = torch.randn(1, 2, 1, 4)  # (B, H, 1, dh)
        nv = torch.randn(1, 2, 1, 4)
        ck, cv = solve(ck, cv, nk, nv)
        assert ck.shape == (1, 2, t + 1, 4), f"第 {t} 步长度错误 {tuple(ck.shape)}"
        assert torch.equal(ck[:, :, t:t + 1], nk), "最新 token 应落在序列末尾"
        if t > 0:
            assert torch.equal(ck[:, :, :t], prev_k), "旧缓存内容不应被改动"
        prev_k, prev_v = ck.clone(), cv.clone()


def check_input_not_mutated(solve):
    ck = torch.randn(1, 2, 3, 4)
    cv = torch.randn(1, 2, 3, 4)
    nk = torch.randn(1, 2, 1, 4)
    nv = torch.randn(1, 2, 1, 4)
    ck0, cv0 = ck.clone(), cv.clone()
    solve(ck, cv, nk, nv)
    assert torch.equal(ck, ck0) and torch.equal(cv, cv0), "不应原地修改缓存张量"


check_basic.example = True
