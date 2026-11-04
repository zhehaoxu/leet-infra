"""SenseVoice rich-token 多任务前缀账——判题用例（order 829）。"""

import torch


def check(solve):
    check_encode_prefix(solve)
    check_rich_ce(solve)
    check_ctc_greedy(solve)
    check_ban_emo(solve)
    check_ctc_shift(solve)
    print("all checks passed")


def check_encode_prefix(solve):
    """前缀构造：查表行号、顺序 [lid, event, emo, textnorm]、映射常量。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    d = 5
    w = torch.randn(16, d)
    out = solve(mode="encode_prefix", embed_w=w, lid=3, textnorm=14)
    assert out["n_added"] == 4
    assert out["prefix"].shape == (4, d)
    assert torch.equal(out["prefix"][0], w[3])   # lid (zh)
    assert torch.equal(out["prefix"][1], w[1])   # event
    assert torch.equal(out["prefix"][2], w[2])   # emo
    assert torch.equal(out["prefix"][3], w[14])  # withitn
    # 行号切换
    out2 = solve(mode="encode_prefix", embed_w=w, lid=13, textnorm=15)
    assert torch.equal(out2["prefix"][0], w[13])
    assert torch.equal(out2["prefix"][3], w[15])


def check_rich_ce(solve):
    """rich CE：eps=0 逐位置对拍、ignore 不计分母、acc、ε=1 均匀项、全 ignore。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    b, d, vocab, ignore = 2, 4, 7, 6
    head = torch.randn(4, b, d)
    ctc_lo = torch.randn(vocab, d) * 0.5
    labels = torch.randint(0, vocab - 1, (4, b))
    labels[0, 1] = ignore
    out = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                labels=labels, ignore_id=ignore, eps=0.0)
    dec = torch.einsum("vd,hbd->hbv", ctc_lo, head)
    nlls = []
    for h in range(4):
        for i in range(b):
            if labels[h, i].item() == ignore:
                continue
            nlls.append(-dec[h, i].log_softmax(-1)[labels[h, i]])
    assert torch.allclose(out["loss"], torch.stack(nlls).mean(), atol=1e-12)
    pred = dec.argmax(-1)
    keep = labels != ignore
    acc_manual = (pred[keep] == labels[keep]).float().mean().item()
    assert abs(out["acc"] - acc_manual) < 1e-12
    # ε=1：loss == 均匀分布交叉熵（只在非 ignore 位置）
    out1 = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                 labels=labels, ignore_id=ignore, eps=1.0)
    logp = dec.log_softmax(-1)
    assert torch.allclose(out1["loss"], (-logp[keep].mean(-1)).mean(),
                          atol=1e-12)
    # ε 混合闭式：(1-ε)·nll_mean + ε·smooth_mean
    out5 = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                 labels=labels, ignore_id=ignore, eps=0.3)
    nll_m = torch.stack(nlls).mean()
    sm_m = (-logp[keep].mean(-1)).mean()
    assert torch.allclose(out5["loss"], 0.7 * nll_m + 0.3 * sm_m, atol=1e-12)
    # 全 ignore：loss=0、acc=0
    labels_ign = torch.full((4, b), ignore)
    out_i = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                  labels=labels_ign, ignore_id=ignore, eps=0.5)
    assert out_i["loss"].item() == 0.0 and out_i["acc"] == 0.0


def check_ctc_greedy(solve):
    """greedy：argmax、unique_consecutive、blank 过滤、frames 保真。"""
    torch.set_default_dtype(torch.float64)
    vocab, blank = 8, 0
    ids = [3, 3, 0, 0, 3, 5, 5, 0, 7]
    logits = torch.full((len(ids), vocab), -1.0)
    for t, i in enumerate(ids):
        logits[t, i] = 5.0
    out = solve(mode="ctc_greedy", logits=logits, blank_id=blank)
    assert out["tokens"] == [3, 3, 5, 7], out["tokens"]
    assert out["frames"] == ids
    # 全 blank → 空 tokens
    lb = torch.full((4, vocab), -1.0)
    lb[:, blank] = 5.0
    out_b = solve(mode="ctc_greedy", logits=lb, blank_id=blank)
    assert out_b["tokens"] == []
    # log_softmax 输入不改变 argmax 语义
    out_sm = solve(mode="ctc_greedy", logits=logits.log_softmax(-1),
                   blank_id=blank)
    assert out_sm["tokens"] == out["tokens"]
    # 非相邻同 id 不折叠
    lg2 = torch.full((3, vocab), -1.0)
    for t, i in enumerate([3, 4, 3]):
        lg2[t, i] = 5.0
    out2 = solve(mode="ctc_greedy", logits=lg2, blank_id=blank)
    assert out2["tokens"] == [3, 4, 3]
    # 空输入
    out_e = solve(mode="ctc_greedy", logits=torch.zeros(0, vocab),
                  blank_id=blank)
    assert out_e["tokens"] == [] and out_e["frames"] == []


def check_ban_emo(solve):
    """ban_emo：-inf 屏蔽、argmax 永不落 unk、changed 掩码语义。"""
    torch.set_default_dtype(torch.float64)
    vocab, emo_unk = 8, 5
    logits = torch.randn(4, vocab)
    out = solve(mode="ban_emo", logits=logits, emo_unk_id=emo_unk)
    assert torch.all(out["masked"][:, emo_unk] == -float("inf"))
    assert torch.equal(out["masked"][:, :emo_unk], logits[:, :emo_unk])
    assert torch.equal(out["masked"][:, emo_unk + 1:], logits[:, emo_unk + 1:])
    assert (out["masked"].argmax(-1) != emo_unk).all()
    before = logits.argmax(-1)
    assert torch.equal(out["changed"], before == emo_unk)
    # before==emo_unk 的帧，屏蔽后取次优（且 changed）
    lg2 = torch.randn(2, vocab)
    lg2[0, emo_unk] = 10.0
    lg2[1, 2] = 10.0
    out2 = solve(mode="ban_emo", logits=lg2, emo_unk_id=emo_unk)
    assert out2["changed"][0].item()
    assert not out2["changed"][1].item()
    assert out2["masked"][0].argmax().item() != emo_unk
    assert out2["masked"][1].argmax().item() == 2
    # 排除 -inf 主导：emo_unk 幅值小于其他帧时 changed 应为 False
    lg3 = torch.randn(1, vocab)
    lg3[0, emo_unk] = -10.0
    out3 = solve(mode="ban_emo", logits=lg3, emo_unk_id=emo_unk)
    assert not out3["changed"][0].item()


def check_ctc_shift(solve):
    """偏移窗：[4, L) 区间（内容 L-4 帧）、len<=4 空窗。"""
    out = solve(mode="ctc_shift", lens=[12, 4, 5, 3])
    assert out["starts"] == [4, 4, 4, 4]
    assert out["ends"] == [12, 4, 5, 4]
    for s, e in zip(out["starts"], out["ends"]):
        assert s <= e
    # 长 12 的条目：内容帧数 8 = 12-4
    assert out["ends"][0] - out["starts"][0] == 8
    # L=5：内容 1 帧；L<4：空窗
    assert out["ends"][2] - out["starts"][2] == 1
    assert out["ends"][3] == out["starts"][3] == 4
