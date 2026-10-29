import torch


def solve(gt_boxes, feature_sizes, strides, scales, ratios, image_size,
          pos_thr=0.5, neg_thr=0.4):
    """多尺度 anchor 生成 + IoU 阈值匹配：语义 = SSD/RetinaNet 的 anchor 目标分配。

    入参：
    - gt_boxes: (G,4) float32/float64 张量，xyxy 绝对坐标；G 可为 0（无 GT）。
    - feature_sizes: 长度 L 的序列，每项 (H, W) 正整数，对应各层特征图尺寸。
    - strides: 长度 L 的正整数序列，各层步长（像素/格）。
    - scales: 正浮点数序列；ratios: 正浮点数序列。
    - image_size: (W_img, H_img) 整数序列，anchor 坐标裁剪边界。
    - pos_thr: 正样本 IoU 阈值（默认 0.5）；neg_thr: 负样本 IoU 阈值（默认 0.4）。
      要求 0 <= neg_thr <= pos_thr，否则 ValueError。

    anchor 生成（顺序：层 → 行 y → 列 x → scale → ratio）：
    中心 (cx,cy)=((x+0.5)*stride,(y+0.5)*stride)；
    每个 (scale,ratio) 给出宽 scale*sqrt(ratio)、高 scale/sqrt(ratio)；
    坐标先成框再裁剪到 [0,W_img]x[0,H_img]（裁剪后可能退化，仍保留）。

    匹配：
    1) 逐 anchor 取 IoU 最大的 GT（并列取 GT 下标最小者）得 best_gt/best_iou；
    2) best_iou >= pos_thr 记正样本（标签 = 匹配到的 GT 下标）；
    3) best_iou <  neg_thr 记负样本（标签 = -1）；
    4) 其余为忽略样本（标签 = -2）；
    5) 保底规则：每个 GT 取「best_gt 指向它」的 anchor 中 IoU 最大者（并列取 anchor
       下标最小者）强制置为正样本，即使其 IoU 低于 pos_thr；该步在 2)-4) 之后执行。
    G=0 时不存在保底步骤，全部 anchor 为负样本。

    IoU 口径：交集面积按裁剪后坐标算，分母为两框面积之和减交集；两框面积均为 0 时
    IoU 记 0。boxes 维度非法、阈值越界、feature_sizes/strides/scales/ratios 长度或
    取值非法、dtype 非 float32/float64 一律 ValueError。

    返回三元组：
    - anchors: (A,4) 与 gt_boxes 同 dtype，A = sum_l H_l*W_l*len(scales)*len(ratios)；
      无 GT 时也要返回完整 anchors。
    - labels: (A,) int64，取值 >=0 表示匹配 GT 下标，-1 负样本，-2 忽略。
    - counts: (3,) int64，依次为 [正样本数, 负样本数, 忽略样本数]，三者之和等于 A。
    """
    # TODO
    raise NotImplementedError
