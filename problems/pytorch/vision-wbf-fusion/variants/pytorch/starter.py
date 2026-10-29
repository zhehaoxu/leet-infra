import torch


def solve(boxes, scores, labels, iou_thr=0.55, num_models=1):
    """加权框融合（Weighted Boxes Fusion）：单模型/多模型检测结果的框级融合。

    入参：
    - boxes: (N,4) float32/float64 张量，xyxy 绝对坐标。
    - scores: (N,) 与 boxes 同 dtype 的置信度；labels: (N,) 整型类别下标。
    - iou_thr: 合并阈值，取值 [0,1]；IoU **严格大于**它才并入已有簇。
    - num_models: 融合来源数（>=1 的整数），用于最终置信度缩放。

    算法（按置信度降序处理，并列取原始下标小者）：
    1) 依次取出每个框，在**同类别**的已有簇中找 IoU 最大的簇；若该最大 IoU **严格大于**
       iou_thr 则并入（并列取创建顺序最早者），否则新建一簇（簇的顺序 = 首次出现顺序）。
       这里的 IoU 是与该簇**当前融合框**的 IoU，且并入后融合框立即重算。
    2) 簇的融合框 = 成员框按置信度**加权平均**（权重即 scores；权重之和为 0 时退化为
       等权平均）。
    3) 簇的融合置信度 = 成员置信度的**算术平均** × min(n, num_models) / num_models，
       其中 n 是簇内框数（对应论文 Eq. 4 的 min(T,N)/N，T 为簇内框数、N 为模型数）。
    4) 输出按融合置信度降序排列，并列时按簇的创建顺序（即首次出现的先后）。

    返回三元组：
    - fused_boxes: (K,4) 与 boxes 同 dtype；
    - fused_scores: (K,) 与 boxes 同 dtype；
    - fused_labels: (K,) int64。
    空输入（N=0）返回三个空张量。

    非法输入一律 ValueError：boxes 非 float32/float64 张量或形状不是 (N,4)、scores/labels
    形状不是 (N,) 或 dtype 不匹配、labels 非整数、iou_thr 不在 [0,1]、num_models < 1。
    """
    # TODO
    raise NotImplementedError
