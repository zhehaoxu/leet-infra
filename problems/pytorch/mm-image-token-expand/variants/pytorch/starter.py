import torch


def solve(text_embeds, labels, attention_mask, image_mask, image_features):
    """单图占位展开与右补齐 / Single-image expansion and right padding.

    中文契约：
    - text_embeds：CPU float32/float64 [B,L,D]；labels：CPU int64 [B,L]；
      attention_mask、image_mask：CPU bool [B,L]。1<=B<=4，1<=L<=16，
      1<=D<=8，输入可以不连续。
    - image_features 是长度 B 的张量列表，第 b 项为 [N_b,D]，1<=N_b<=12，
      与 text_embeds 同 dtype/device，也可以不连续。
    - attention_mask 支持左 padding、右 padding、内部空洞。每行有效位置
      恰有一个 image_mask=True；标记数不为 1 或无效位置有图像标记时抛出
      ValueError。其余输入保证合法。
    - 按原顺序丢弃无效位置，将单个有效图像占位替换成该行全部视觉 embedding。
      有效文本 embedding/label 原样保留；视觉 labels 为 -100。不 shift labels，
      不截断，不查词表或调用模型权重；不得修改任何输入或列表。
    - 返回字典，且恰有以下五个 key，S=max_b(有效长度_b-1+N_b)：
      inputs_embeds：[B,S,D]，原 dtype；右侧补 0。
      labels：int64 [B,S]；视觉位置和右侧 padding 为 -100。
      attention_mask：bool [B,S]；有效位置 True，右侧 padding False。
      position_ids：int64 [B,S]；每行有效位置 0..长度-1，padding 为 0。
      mm_token_type_ids：int64 [B,S]；视觉位置 1，文本及 padding 为 0。
      所有输出均在 CPU；文本 label=-100 不代表该位置是图像。

    English contract:
    - text_embeds is CPU float32/float64 [B,L,D]; labels is CPU int64 [B,L].
      attention_mask and image_mask are CPU bool [B,L]. Bounds: B=1..4,
      L=1..16, D=1..8. Inputs may be noncontiguous.
    - image_features is a length-B tensor list; item b is [N_b,D], N_b=1..12,
      with text_embeds' dtype/device, and may be noncontiguous.
    - Attention may contain left/right padding or internal holes. Each row
      must have exactly one image marker at a valid position. Raise ValueError
      for any other marker count or any marker in padding. Other inputs are valid.
    - Remove invalid positions in order, then replace the one image placeholder
      with that row's visual embeddings. Preserve valid text embeddings/labels;
      all visual labels become -100. Do not shift labels, truncate, perform
      embedding lookup, use model weights, or mutate any input (including the list).
    - Return exactly five dictionary keys, all on CPU. Let
      S=max_b(valid_length_b-1+N_b), and right-pad every row to S:
      inputs_embeds: [B,S,D], original dtype, padding 0;
      labels: int64 [B,S], visual/padding values -100;
      attention_mask: bool [B,S], valid True, padding False;
      position_ids: int64 [B,S], valid positions 0..length-1, padding 0;
      mm_token_type_ids: int64 [B,S], image 1, text/padding 0.
      A text label of -100 does not make that token an image.
    """
    raise NotImplementedError
