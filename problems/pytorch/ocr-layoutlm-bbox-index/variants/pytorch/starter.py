def solve(boxes, page_w, page_h):
    """LayoutLM 家族 bbox 归一化与 2D 位置索引账（HF `_calc_spatial_position_embeddings` 口径）。

    输入
    ----
    boxes  : list[[x0,y0,x1,y1]] 像素坐标（x0<=x1, y0<=y1）
    page_w : 页宽像素（>0，用于 x 归一化）
    page_h : 页高像素（>0，用于 y 归一化）

    记账（题面钉死）：
      1. 归一化到 0–1000：x 方向除以 page_w、y 方向除以 page_h，乘 1000 后
         round（Python banker's rounding），再 clamp 到 [0, 1000]；
      2. 宽高嵌入 id：w_id = x1 - x0、h_id = y1 - y0（归一化后，官方
         `bbox[:,:,2]-bbox[:,:,0]` 与 `bbox[:,:,3]-bbox[:,:,1]`）；
      3. 2D 网格索引：grid(x, y) = y * 1001 + x（0..1000 共 1001 列）；
         每 box 输出四角网格 id [g00, g01, g10, g11] =
         [g(x0,y0), g(x1,y0), g(x0,y1), g(x1,y1)]，及中心
         cx=(x0+x1)//2、cy=(y0+y1)//2 的 center 网格 id。

    返回
    ----
    dict：
      norm_boxes    list[[int]*4]  归一化后四角
      wh_ids        list[[int,int]] 每个 box 的 [w_id, h_id]
      corner_grid   list[[int]*4]   四角网格 id
      center_grid   list[int]       中心网格 id
    """
    raise NotImplementedError
