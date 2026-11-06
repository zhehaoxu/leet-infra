"""Qwen3-ASR AuT 分块编码器账（回源：QwenLM/Qwen3-ASR modeling_qwen3_asr.py）。

官方 Qwen3-ASR 的 AuT 音频编码器核心设计：**mel 帧先按固定窗分块（chunk），
每个 chunk 独立过三层 Conv2d（8× 下采样）**，因此卷积的边界 padding 每个
chunk 各吃一次——下采样长度公式直接把分块写死（`leave = lens % 100` 单独
算、满 chunk 每个 ×13 tokens）；chunk 级位置编码每块从 0 重新开始；注意力
按 cu_seqlens 分窗（段内双向、段间隔离，块对角）。本题重建其中可判定的账：

1. **per-chunk 下采样长度**（cnn_lengths）：`_get_feat_extract_output_lengths`
   ——`leave = lens % 100`（硬编码 100，不随 config 走）、对 leave 单独做三次
   `(l-1)//2+1`、再加满 chunk 数 `(lens // 100) * 13`（每个 100 帧满 chunk
   独立 conv 恰产 13 tokens：50→25→13）。
2. **分块计划**（chunk_plan）：`chunk_num = ceil(lens / chunk_frames)`、
   chunk_lengths 全满窗、尾 chunk 按各样本 `lens % chunk_frames` 覆盖、
   **整除时余 0 回落满窗**（quirk：`chunk_lengths[==0] = chunk_frames`）、
   tail_chunk_index 用 `F.pad(chunk_num, (1,0), -1).cumsum(0)[1:]`。
3. **串切打包**（pack_chunks）：**batch 内多样本的 mel 先串接再统一切块**
   （块是跨样本的独立单元），pad_sequence 对齐成 (N, C, Tmax)、CNN 后每块
   长度 `_get_feat_extract_output_lengths(chunk_lengths)`、bool mask 记有效位。
4. **位置编码与展平**（pos_flatten）：正弦位置表截前 Tmax 行**逐块相加**
   （每块位置从 0 起，pad 位置也加、随后被 mask 丢弃），
   `padded_embed[mask]` 布尔展平串回一条序列。
5. **窗口分窗**（cu_windows）：`window_aftercnn = padded_t * (infer // chunk)`
   ——窗口长度由 **padded 后**的块长推导；每样本按 aftercnn_len 分满窗 +
   余数窗，cu_chunk_lens 全局 cumsum 成 cu_seqlens。
6. **块对角注意力**（block_attn）：eager 路径——q/k/v 权重显式入参，
   cu_seqlens 各段内标准 softmax(QKᵀ/√d)V 双向注意力，**段间完全不往来**
   （等价官方 4D mask 的块对角布局）。

所有张量显式入参，fp64；时间维账纯整数。MRoPE 交错布局已被 omni 合集占，
不入题。

判题约定：单入口 solve(mode, **kw)；CHECKS 列出全部校验函数。
"""

import math

import torch
from torch import Tensor

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# 官方常量（HF config 默认 / 源码硬编码）
CHUNK_FRAMES = 100       # n_window * 2（权重实际 n_window=50）；下采样公式里硬编码 100
INFER_WINDOW = 800       # n_window_infer（推理期固定 8s @ 100Hz）
TOKENS_PER_CHUNK = 13    # 每个 100 帧满 chunk 独立 conv 的输出 token 数


def _conv3_len(l: int) -> int:
    """单层 Conv2d(kernel 3, stride 2, pad 1) 的时间维输出长度。"""
    return (l - 1) // 2 + 1


def solve(mode: str, **kw):
    if mode == "cnn_lengths":
        lens = kw["feature_lens"]
        # 官方公式：leave 单独三次下采样 + 满 chunk 数 × 13（硬编码 100/13）
        leave = lens % CHUNK_FRAMES
        l1 = (leave - 1) // 2 + 1
        l2 = (l1 - 1) // 2 + 1
        l3 = (l2 - 1) // 2 + 1
        out = l3 + (lens // CHUNK_FRAMES) * TOKENS_PER_CHUNK
        return {"out": out, "steps": torch.stack([l1, l2, l3], dim=1)}

    if mode == "chunk_plan":
        lens = kw["feature_lens"]
        chunk_frames = kw.get("chunk_frames", CHUNK_FRAMES)
        chunk_num = torch.ceil(lens.to(torch.float64) / chunk_frames).long()
        chunk_lengths = torch.tensor(
            [chunk_frames] * int(chunk_num.sum()), dtype=torch.long
        )
        tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = lens % chunk_frames
        chunk_lengths[chunk_lengths == 0] = chunk_frames   # 整除 quirk：余 0 回落满窗
        return {
            "chunk_num": chunk_num,
            "chunk_lengths": chunk_lengths,
            "tail_chunk_index": tail_chunk_index,
        }

    if mode == "pack_chunks":
        features = kw["features"]          # (C, T) 单样本 mel（batch 混排同理先串接）
        lens = kw["feature_lens"]
        chunk_frames = kw.get("chunk_frames", CHUNK_FRAMES)
        plan = solve("chunk_plan", feature_lens=lens, chunk_frames=chunk_frames)
        chunk_lengths = plan["chunk_lengths"]
        chunk_list = features.T.split(chunk_lengths.tolist(), dim=0)
        padded = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)  # (N, Tmax, C)
        padded = padded.transpose(1, 2)                                         # (N, C, Tmax)
        after_cnn = solve("cnn_lengths", feature_lens=chunk_lengths)["out"]
        mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(int(l), dtype=torch.bool) for l in after_cnn], batch_first=True
        )
        return {
            "padded": padded,
            "after_cnn": after_cnn,
            "mask": mask,
            "n_chunks": int(chunk_num := plan["chunk_num"].sum()),
        }

    if mode == "pos_flatten":
        padded_embed = kw["padded_embed"]  # (N, Tmax, D) conv_out 之后
        mask = kw["mask"]                  # (N, Tmax) bool
        pos_emb = kw["pos_emb"]            # (Tmax, D) 正弦位置表
        tmax = padded_embed.shape[1]
        # 位置表截前 Tmax 行，逐块相加（每块位置从 0 起；pad 位也加、随后被 mask 丢）
        hidden = padded_embed + pos_emb[:tmax].unsqueeze(0)
        flat = hidden[mask]                # 布尔展平：各块有效位串回一条序列
        return {"hidden": hidden, "flat": flat}

    if mode == "cu_windows":
        aftercnn_lens = kw["aftercnn_lens"]
        padded_t = kw["padded_t"]
        infer_window = kw.get("infer_window", INFER_WINDOW)
        chunk_frames = kw.get("chunk_frames", CHUNK_FRAMES)
        window_aftercnn = padded_t * (infer_window // chunk_frames)
        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens).cumsum(-1, dtype=torch.int32)
        return {
            "window": window_aftercnn,
            "cu_chunk_lens": cu_chunk_lens,
            "cu_seqlens": cu_seqlens,
        }

    if mode == "block_attn":
        hidden = kw["hidden"]              # (S, D)
        cu_seqlens = kw["cu_seqlens"]
        wq = kw["wq"]; wk = kw["wk"]; wv = kw["wv"]; wo = kw["wo"]
        scaling = kw.get("scaling", None)
        S, D = hidden.shape
        if scaling is None:
            scaling = D ** -0.5
        q = (hidden @ wq) * scaling        # (S, D)
        k = hidden @ wk
        v = hidden @ wv
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist()
        for a, b in zip(bounds[:-1], bounds[1:]):
            qs, ks, vs = q[a:b], k[a:b], v[a:b]           # 段内双向、段间隔离
            attn = torch.softmax(qs @ ks.T, dim=-1)
            out[a:b] = attn @ vs
        return {"out": out @ wo}

    raise ValueError(f"unknown mode: {mode}")
