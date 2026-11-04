"""SenseVoice rich-token 多任务前缀账（回源：FunAudioLLM/SenseVoice model.py）。

官方 SenseVoiceSmall 的核心设计：**CTC 输入前拼 4 帧 rich-token query**，前 4 帧
经 ctc_lo 的预测算 rich CE（语言/事件/情绪/文本规范四头），第 5 帧起才是真正的
CTC 语音内容（`encoder_out[:, 4:]` + `lengths - 4`）。本题重建其中可判定的账：

1. **前缀查询构造**（encode_prefix）：16 行小 embed 表（0=auto, 1=event,
   2=emo, 3/4/7/11/12/13=各语言, 14/15=withitn/woitn），inference 顺序
   `[lid, event, emo, textnorm]`（官方代码先 cat(textnorm) 再 cat(input_query)
   ——最终序列前缀是 [lid, event, emo, textnorm]，训练 encode 同序），
   speech_lengths += 4。
2. **rich CE**（rich_ce）：`_calc_rich_ce_loss`——前 4 帧 encoder 输出经 ctc_lo
   投影到 vocab，对标签算 LabelSmoothingLoss（ignore_id 跳过、ε 均匀项），
   另出 th_accuracy（ignore_label 不计分母）。
3. **CTC greedy**（ctc_greedy）：log_softmax → 逐帧 argmax →
   unique_consecutive（同 id 连帧折叠）→ 过滤 blank。
4. **ban_emo_unk**（ban_emo）：情绪 unk 位置置 -inf 后，该位置永不被 argmax
   选中（仅当该帧原本就 argmax 为它时才改变结果）。
5. **CTC 偏移窗**（ctc_shift）：内容 CTC 的切片账——`encoder_out[:, 4:]`、
   `lens - 4`；len<=4 时窗为空。

所有张量显式入参，fp64，无 RNG（训练侧 lid 90% 采样不进判题域）。

判题约定：单入口 solve(mode, **kw)；CHECKS 列出全部校验函数。
"""

import torch
from torch import Tensor

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 官方词表常量（model.py）
LID_INT_DICT = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
TEXTNORM_INT_DICT = {25016: 14, 25017: 15}
EMO_DICT = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003,
            "neutral": 25004}
NUM_PREFIX = 4


def _label_smoothing_loss(logits: Tensor, target: Tensor, ignore_id: int,
                          eps: float) -> Tensor:
    """官方 LabelSmoothingLoss 语义（Kaldi 风格，normalize_over_non_ignored）：
    nll 与均匀项都只在非 ignore 位置累计，除以非 ignore 位置数。"""
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab)
    tgt = target.reshape(-1)
    keep = tgt != ignore_id
    if not keep.any():
        return flat.sum() * 0.0
    logp = flat.log_softmax(-1)
    nll = -logp[keep, tgt[keep]]
    smooth = -logp[keep].mean(-1)  # 均匀分布交叉熵 = -mean(logp)
    return ((1.0 - eps) * nll + eps * smooth).mean()


def _th_accuracy(logits: Tensor, target: Tensor, ignore_id: int) -> float:
    """官方 th_accuracy：逐位置 argmax 命中率，ignore 位置不计分母。"""
    pred = logits.reshape(-1, logits.shape[-1]).argmax(-1)
    tgt = target.reshape(-1)
    keep = tgt != ignore_id
    if not keep.any():
        return 0.0
    return (pred[keep] == tgt[keep]).float().mean().item()


def solve(mode: str, **kw):
    """SenseVoice rich-token 账。各 mode 参数与返回值见分支 docstring。"""
    if mode == "encode_prefix":
        """前缀查询构造。kw: embed_w (16, d), lid (int embed 行号 0..13),
        textnorm (int 14|15)。返回 {"prefix", "n_added"}：4 帧前缀
        [lid, event, emo, textnorm]（各 (1, d)）与新增帧数 4。
        官方 inference/encode 的 query 拼接账。"""
        w = kw["embed_w"]
        idx = torch.tensor([kw["lid"], 1, 2, kw["textnorm"]], dtype=torch.long)
        prefix = w[idx]  # (4, d)
        return {"prefix": prefix, "n_added": NUM_PREFIX}

    if mode == "rich_ce":
        """rich CE + 准确率。kw: encoder_head (4, b, d), ctc_lo (vocab, d),
        labels (4, b)（ctc 词表 id，ignore_id 跳过）, ignore_id, eps
        返回 {"loss", "acc"}。官方 _calc_rich_ce_loss。"""
        head = kw["encoder_head"]  # (4, b, d)
        vocab, d = kw["ctc_lo"].shape
        decoder_out = torch.einsum("vd,hbd->hbv", kw["ctc_lo"], head)
        loss = _label_smoothing_loss(decoder_out, kw["labels"],
                                     kw["ignore_id"], kw["eps"])
        acc = _th_accuracy(decoder_out, kw["labels"], kw["ignore_id"])
        return {"loss": loss, "acc": acc}

    if mode == "ctc_greedy":
        """CTC greedy 解码。kw: logits (n, vocab)（未归一化）, blank_id
        返回 {"tokens", "frames"}：折叠后非 blank token id 与对应 argmax 帧
        序列（unique_consecutive 之前）。官方 inference 的 greedy 段。"""
        logits = kw["logits"]
        yseq = logits.argmax(dim=-1)  # (n,)
        uniq = torch.unique_consecutive(yseq)
        mask = uniq != kw["blank_id"]
        return {"tokens": uniq[mask].tolist(), "frames": yseq.tolist()}

    if mode == "ban_emo":
        """情绪 unk 屏蔽。kw: logits (n, vocab), emo_unk_id
        返回 {"masked", "changed"}：置 -inf 后的 logits 与该操作实际改变
        argmax 结果的帧 bool 掩码。官方 ban_emo_unk。"""
        logits = kw["logits"]
        before = logits.argmax(-1)
        masked = logits.clone()
        masked[..., kw["emo_unk_id"]] = -float("inf")
        after = masked.argmax(-1)
        changed = (before != after) | (before == kw["emo_unk_id"])
        return {"masked": masked, "changed": changed}

    if mode == "ctc_shift":
        """内容 CTC 偏移窗。kw: lens (list[int])（含 4 帧前缀的总长）
        返回 {"starts", "ends"}：内容帧区间 [4, lens)；官方
        `encoder_out[:, 4:]` 配 `lens - 4` 即前 lens-4 帧。
        len<=4 时为空窗 (start==end==4)。"""
        starts, ends = [], []
        for L in kw["lens"]:
            s = NUM_PREFIX
            e = max(NUM_PREFIX, int(L))
            starts.append(s)
            ends.append(e)
        return {"starts": starts, "ends": ends}

    raise ValueError(f"unknown mode: {mode}")


@check
def check_encode_prefix(solve):
    """前缀构造：查表行号、顺序 [lid, event, emo, textnorm]、形状。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    d = 5
    w = torch.randn(16, d)
    out = solve(mode="encode_prefix", embed_w=w, lid=3, textnorm=14)
    assert out["n_added"] == 4
    assert out["prefix"].shape == (4, d)
    assert torch.equal(out["prefix"][0], w[3])   # lid
    assert torch.equal(out["prefix"][1], w[1])   # event
    assert torch.equal(out["prefix"][2], w[2])   # emo
    assert torch.equal(out["prefix"][3], w[14])  # textnorm
    # 词表映射账
    assert LID_INT_DICT[24884] == 3 and TEXTNORM_INT_DICT[25016] == 14
    assert EMO_DICT["unk"] == 25009
    out2 = solve(mode="encode_prefix", embed_w=w, lid=13, textnorm=15)
    assert torch.equal(out2["prefix"][0], w[13])
    assert torch.equal(out2["prefix"][3], w[15])


@check
def check_rich_ce(solve):
    """rich CE：eps=0 时逐位置 CE 对拍、ignore 不计分母、acc、ε 均匀项。"""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1)
    b, d, vocab, ignore = 2, 4, 7, 6
    head = torch.randn(4, b, d)
    ctc_lo = torch.randn(vocab, d) * 0.5
    labels = torch.randint(0, vocab - 1, (4, b))
    labels[0, 1] = ignore
    out = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                labels=labels, ignore_id=ignore, eps=0.0)
    # 逐位置对拍（eps=0）
    dec = torch.einsum("vd,hbd->hbv", ctc_lo, head)
    nlls = []
    for h in range(4):
        for i in range(b):
            if labels[h, i].item() == ignore:
                continue
            nlls.append(-dec[h, i].log_softmax(-1)[labels[h, i]])
    assert torch.allclose(out["loss"], torch.stack(nlls).mean(), atol=1e-12)
    # acc：ignore 位置不计
    pred = dec.argmax(-1)
    keep = labels != ignore
    acc_manual = (pred[keep] == labels[keep]).float().mean().item()
    assert abs(out["acc"] - acc_manual) < 1e-12
    # ε 均匀项：eps=1 时 loss == -mean logp（全位置均匀交叉熵均值）
    out1 = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                 labels=labels, ignore_id=ignore, eps=1.0)
    logp = dec.log_softmax(-1)
    smooth_manual = (-logp[keep].mean(-1)).mean()
    assert torch.allclose(out1["loss"], smooth_manual, atol=1e-12)
    # 全 ignore → loss 0、acc 0
    labels_ign = torch.full((4, b), ignore)
    out_i = solve(mode="rich_ce", encoder_head=head, ctc_lo=ctc_lo,
                  labels=labels_ign, ignore_id=ignore, eps=0.5)
    assert out_i["loss"].item() == 0.0 and out_i["acc"] == 0.0


@check
def check_ctc_greedy(solve):
    """greedy：argmax、unique_consecutive 折叠、blank 过滤、frames 保真。"""
    torch.set_default_dtype(torch.float64)
    vocab, blank = 8, 0
    # 构造 argmax 帧序列 [3,3,0,0,3,5,5,0,7] → 折叠 [3,0,3,5,0,7] → 去重后非 blank [3,3,5,7]
    ids = [3, 3, 0, 0, 3, 5, 5, 0, 7]
    logits = torch.full((len(ids), vocab), -1.0)
    for t, i in enumerate(ids):
        logits[t, i] = 5.0
    out = solve(mode="ctc_greedy", logits=logits, blank_id=blank)
    assert out["tokens"] == [3, 3, 5, 7], out["tokens"]
    assert out["frames"] == ids
    # 全 blank → 空
    lb = torch.full((4, vocab), -1.0)
    lb[:, blank] = 5.0
    out_b = solve(mode="ctc_greedy", logits=lb, blank_id=blank)
    assert out_b["tokens"] == []
    # 纯 log_softmax 语义：logits 非负化不改变 argmax
    out_sm = solve(mode="ctc_greedy", logits=logits.log_softmax(-1),
                   blank_id=blank)
    assert out_sm["tokens"] == out["tokens"]
    # 同 id 不相邻不折叠
    ids2 = [3, 4, 3]
    lg2 = torch.full((3, vocab), -1.0)
    for t, i in enumerate(ids2):
        lg2[t, i] = 5.0
    out2 = solve(mode="ctc_greedy", logits=lg2, blank_id=blank)
    assert out2["tokens"] == [3, 4, 3]


@check
def check_ban_emo(solve):
    """ban_emo：-inf 屏蔽后 argmax 语义与 changed 掩码。"""
    torch.set_default_dtype(torch.float64)
    vocab, emo_unk = 8, 5
    logits = torch.randn(4, vocab)
    out = solve(mode="ban_emo", logits=logits, emo_unk_id=emo_unk)
    assert torch.all(out["masked"][:, emo_unk] == -float("inf"))
    assert torch.equal(out["masked"][:, :emo_unk], logits[:, :emo_unk])
    assert torch.equal(out["masked"][:, emo_unk + 1:], logits[:, emo_unk + 1:])
    # argmax 永不落在 emo_unk
    assert (out["masked"].argmax(-1) != emo_unk).all()
    # changed：屏蔽前 argmax==emo_unk 的帧才可能 changed
    before = logits.argmax(-1)
    expect_changed = (before == emo_unk)
    assert torch.equal(out["changed"], expect_changed)
    # 若某帧 before==emo_unk 且次优唯一，changed 后取次优
    lg2 = torch.randn(1, vocab)
    lg2[0, emo_unk] = 10.0
    out2 = solve(mode="ban_emo", logits=lg2, emo_unk_id=emo_unk)
    assert out2["changed"][0].item() is True or out2["changed"][0].item()
    assert out2["masked"][0].argmax().item() != emo_unk


@check
def check_ctc_shift(solve):
    """偏移窗：[4, L-4) 区间、len<=4 空窗。"""
    out = solve(mode="ctc_shift", lens=[12, 4, 5, 3])
    assert out["starts"] == [4, 4, 4, 4]
    assert out["ends"] == [8, 4, 4, 4]
    # 内容帧数 = L - 4（非负）
    for s, e in zip(out["starts"], out["ends"]):
        assert s <= e


def check(solve):
    for fn in CHECKS:
        fn(solve)
