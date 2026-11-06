"""Qwen3-ASR AuT 分块编码器账——判题用例（order 830）。"""

import torch


def check(solve):
    check_cnn_lengths(solve)
    check_chunk_plan(solve)
    check_pack_chunks(solve)
    check_pos_flatten(solve)
    check_cu_windows(solve)
    check_block_attn(solve)
    print("all checks passed")


def check_cnn_lengths(solve):
    """per-chunk 下采样账：%100 硬编码、满 chunk ×13、leave 边界。"""
    torch.set_default_dtype(torch.float64)
    lens = torch.tensor([100, 250, 200, 37, 1], dtype=torch.long)
    out = solve(mode="cnn_lengths", feature_lens=lens)["out"]
    # 手算锚点：leave = l % 100 三次 (l-1)//2+1，再加 (l // 100) * 13
    # 100: leave=0 → 0 → out=13；250: leave=50 → 25→13→7 → 7+26=33
    # 200: leave=0 → 0 → 26；37: 19→10→5 → 5；1: 1→1→1 → 1
    assert out.tolist() == [13, 33, 26, 5, 1]
    steps = solve(mode="cnn_lengths", feature_lens=lens)["steps"]
    assert steps[1].tolist() == [25, 13, 7]      # 250 的逐层账
    assert steps[3].tolist() == [19, 10, 5]      # 37 的逐层账
    assert steps[0].tolist() == [0, 0, 0]        # leave=0：三次 (−1)//2+1 = 0
    # 整除满 chunk：100 的倍数 leave=0，全靠 (l//100)*13 顶
    assert solve(mode="cnn_lengths", feature_lens=torch.tensor([300]))["out"].item() == 39
    # 满 chunk 独立 conv 恰产 13：50→25→13
    assert solve(mode="cnn_lengths", feature_lens=torch.tensor([100]))["out"].item() == 13


def check_chunk_plan(solve):
    """分块计划：ceil 块数、尾块余数覆盖、整除余 0 回落满窗。"""
    torch.set_default_dtype(torch.float64)
    lens = torch.tensor([100, 250], dtype=torch.long)
    plan = solve(mode="chunk_plan", feature_lens=lens)
    assert plan["chunk_num"].tolist() == [1, 3]
    # tail_chunk_index = pad(-1).cumsum()[1:] = [0, 3]；尾块余数 [0, 50]
    assert plan["tail_chunk_index"].tolist() == [0, 3]
    # 尾块 0 回落满窗：[100, 100, 100, 50]
    assert plan["chunk_lengths"].tolist() == [100, 100, 100, 50]
    # 纯整除样本：50 → 单块 [50]；200 整除 → [100, 100]（余 0 回落）
    assert solve(mode="chunk_plan", feature_lens=torch.tensor([50]))["chunk_lengths"].tolist() == [50]
    assert solve(mode="chunk_plan", feature_lens=torch.tensor([200]))["chunk_lengths"].tolist() == [100, 100]
    # chunk_frames 可入参：chunk_frames=40, lens=88 → ceil=3，尾块 8
    plan2 = solve(mode="chunk_plan", feature_lens=torch.tensor([88]), chunk_frames=40)
    assert plan2["chunk_lengths"].tolist() == [40, 40, 8]


def check_pack_chunks(solve):
    """串切打包：切分布局、pad 置零、CNN 后 mask 与长度联动。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(3)
    c, t = 3, 250
    features = torch.randn(c, t)
    res = solve(mode="pack_chunks", features=features, feature_lens=torch.tensor([250]))
    # (100, 100, 50) 三块 → padded (3, 3, 100)
    assert res["padded"].shape == (3, c, 100)
    assert res["after_cnn"].tolist() == [13, 13, 7]
    assert res["n_chunks"] == 3
    # 块内容 == features 切片转置；pad 位严格 0
    assert torch.equal(res["padded"][0], features[:, :100])
    assert torch.equal(res["padded"][1], features[:, 100:200])
    assert torch.equal(res["padded"][2, :, :50], features[:, 200:])
    assert torch.equal(res["padded"][2, :, 50:], torch.zeros(c, 50))
    # mask 行长 13/13/7，总数 33 与 cnn_lengths 联动
    assert res["mask"].sum().item() == 33
    assert res["mask"][0].sum().item() == 13 and res["mask"][2].sum().item() == 7
    # chunk_frames 可入参且 after_cnn 走 per-chunk 账（复用 chunk_plan + cnn_lengths）：
    # 250 帧 chunk_frames=250 → 单块 [250]，per-chunk 账 25→13→7 + 2×13 = 33
    # （连续整体下采样是 125→63→32→16，分叉点在块长 >100 非整百）
    features2 = torch.randn(c, 250)
    res2 = solve(mode="pack_chunks", features=features2,
                 feature_lens=torch.tensor([250]), chunk_frames=250)
    assert res2["padded"].shape == (1, c, 250)
    assert res2["after_cnn"].tolist() == [33]
    assert res2["mask"].sum().item() == 33


def check_pos_flatten(solve):
    """位置编码与展平：每块从 0 起、pad 位也加随后被 mask 丢、布爭展平次序。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(11)
    n, tmax, d = 2, 4, 3
    padded = torch.randn(n, tmax, d)
    pos_emb = torch.randn(tmax + 1, d)   # 多给一行：偏移类变异死于语义而非切片越界
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    res = solve(mode="pos_flatten", padded_embed=padded, mask=mask, pos_emb=pos_emb)
    hidden = res["hidden"]
    # 每块位置从 0 起：块 i 的 t 位加 pos[t]；pad 位（t=2,3）也加、随后被 mask 丢
    assert torch.equal(hidden[0, 2], padded[0, 2] + pos_emb[2])
    assert torch.equal(hidden[1, 3], padded[1, 3] + pos_emb[3])
    assert torch.equal(hidden[0, 0], padded[0, 0] + pos_emb[0])
    # 布尔展平：行 0 的 2 位 + 行 1 的 1 位，次序按 (行, t) 字典序
    flat = res["flat"]
    assert flat.shape == (3, d)
    assert torch.equal(flat[0], padded[0, 0] + pos_emb[0])
    assert torch.equal(flat[1], padded[0, 1] + pos_emb[1])
    assert torch.equal(flat[2], padded[1, 0] + pos_emb[0])
    # 展平长度 == mask.sum()
    assert flat.shape[0] == int(mask.sum())


def check_cu_windows(solve):
    """窗口分窗：window 由 padded 后块长推导、满窗整除无余数窗、全局 cumsum。"""
    torch.set_default_dtype(torch.float64)
    # padded_t=13（pad 后块长），infer=800, chunk=100 → window = 13 * 8 = 104
    res = solve(mode="cu_windows", aftercnn_lens=torch.tensor([26, 7]), padded_t=13)
    assert res["window"] == 104
    # 26、7 都不足一窗 → 各一窗；cu_chunk_lens [0, 26, 7] → cumsum [0, 26, 33]
    assert res["cu_chunk_lens"] == [0, 26, 7]
    assert res["cu_seqlens"].tolist() == [0, 26, 33]
    # 满窗整除：104 恰一窗（余 0 无余数窗）；105 → 104 + 1
    res2 = solve(mode="cu_windows", aftercnn_lens=torch.tensor([104, 105]), padded_t=13)
    assert res2["cu_chunk_lens"] == [0, 104, 104, 1]
    assert res2["cu_seqlens"].tolist() == [0, 104, 208, 209]
    # window 随 padded_t 与入参走：padded_t=10, infer=400 → 10 * 4 = 40
    res3 = solve(mode="cu_windows", aftercnn_lens=torch.tensor([45]), padded_t=10,
                 infer_window=400)
    assert res3["window"] == 40
    assert res3["cu_chunk_lens"] == [0, 40, 5]


def check_block_attn(solve):
    """块对角注意力：段内双向 softmax(QKᵀ/√d)V、段间完全隔离。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(7)
    s, d = 6, 4
    hidden = torch.randn(s, d)
    wq = torch.randn(d, d)
    wk = torch.randn(d, d)
    wv = torch.randn(d, d)
    wo = torch.randn(d, d)
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)
    res = solve(mode="block_attn", hidden=hidden, cu_seqlens=cu,
                wq=wq, wk=wk, wv=wv, wo=wo)
    # oracle 独立重写：段内双向注意力（窗口 [0,2) 与 [2,6)）
    expect = torch.empty(s, d)
    for a, b in [(0, 2), (2, 6)]:
        qs = (hidden[a:b] @ wq) * d ** -0.5
        ks = hidden[a:b] @ wk
        vs = hidden[a:b] @ wv
        expect[a:b] = (torch.softmax(qs @ ks.T, dim=-1) @ vs) @ wo
    assert torch.allclose(res["out"], expect, atol=1e-12)
    # 段间隔离：改第二段 hidden 不影响第一段输出
    hidden2 = hidden.clone()
    hidden2[2:] += 5.0
    res2 = solve(mode="block_attn", hidden=hidden2, cu_seqlens=cu,
                 wq=wq, wk=wk, wv=wv, wo=wo)
    assert torch.equal(res2["out"][:2], res["out"][:2])
    # cu_seqlens 必须真分窗：全序列单窗 [0,6] 与两窗结果不同
    res3 = solve(mode="block_attn", hidden=hidden, cu_seqlens=torch.tensor([0, 6], dtype=torch.int32),
                 wq=wq, wk=wk, wv=wv, wo=wo)
    assert not torch.allclose(res3["out"][:2], res["out"][:2], atol=1e-9)
