import torch


def solve(gt_boxes, points, levels, strides, regress_ranges, center_radius=1.5):
    """FCOS 式逐点目标分配 + center-ness 软标签。

    入参：
    - gt_boxes: (G,4) float32/float64 张量，xyxy 绝对像素坐标；G 可为 0。
    - points: (P,2) 与 gt_boxes 同 dtype，每行是一处采样点的 (x, y) 像素坐标
      （通常取自各层特征图的格心）。
    - levels: 长度 P 的 int64 张量/序列，每点所属的层下标，取值 [0, L)。
    - strides: 长度 L 的正数序列，各层步长（像素/格）。
    - regress_ranges: 长度 L 的序列，每项 (lo, hi)，该层负责的目标尺度区间，
      判定用 max(l,t,r,b)，要求 lo <= max < hi（左闭右开）。
    - center_radius: 不小于 0 的浮点数；中心采样半径 = center_radius * 该点所在层步长。

    逐点定义（对每个采样点 p=(x,y) 与每个 GT g=(x1,y1,x2,y2)）：
    - 距离 l=x-x1, t=y-y1, r=x2-x, b=y2-y1（可为负，表示点在框外）；
    - inside(g)     : l>=0 且 t>=0 且 r>=0 且 b>=0；
    - in_center(g)  : |x-(x1+x2)/2| <= radius 且 |y-(y1+y2)/2| <= radius；
    - in_range(g)   : lo <= max(l,t,r,b) < hi（lo/hi 取该点所在层的区间），
                      注意 max 用的是**该 GT 的**四个距离；
    - 候选(g)       : inside(g) 且 in_center(g) 且 in_range(g)。

    分配规则：
    - 正样本：若该点存在候选 GT，取其中**面积最小**者（面积 = clip(x2-x1,0)*clip(y2-y1,0)，
      并列取 GT 下标最小者），标签 = 该 GT 下标；
    - 忽略样本（-2）：没有候选，但该点落在至少一个 GT 框内；
    - 负样本（-1）：没有任何 GT 包含该点；
    - 无 GT（G=0）时全部为负样本。

    正样本的 center-ness 软标签：
        c = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
    其中 l,t,r,b 是相对**被分配 GT** 的距离；任一 max 为 0 时对应因子取 0
    （例如点正好落在框边上）。非正样本的 center-ness 恒为 0。

    返回三元组：
    - labels: (P,) int64，取值为 GT 下标、-1（负）或 -2（忽略）；
    - centerness: (P,) 与 gt_boxes 同 dtype，正样本为上式，其余为 0；
    - reg_targets: (P,4) 与 gt_boxes 同 dtype，正样本为 [l,t,r,b]（被分配 GT），
      非正样本全 0。

    非法输入一律 ValueError：gt_boxes 非 float32/float64 或形状不是 (G,4)、points 形状不是
    (P,2) 或 dtype 不一致、levels 长度不等于 P 或越界、strides/regress_ranges 长度不一致
    或含非正 stride、区间不满足 lo < hi、center_radius 为负。
    """
    # TODO
    raise NotImplementedError
