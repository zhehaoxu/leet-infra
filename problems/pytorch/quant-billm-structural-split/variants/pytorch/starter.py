import torch


def solve(W, up_lim):
    """BiLLM：按列分组做结构化拆分，组内用逐行残差二值重建。

    `W` 是 [O, C] 权重；`up_lim` 是结构搜索的列数上界（2 <= up_lim <= C）。

    记 `hor(x, mask, order)` 为逐行残差二值：对每一行只在 `mask` 为真的列上，
    先取均值 `mean`，再减均值、取 `scale = mean(|centered|)`，
    重建值为 `sign(centered) * scale + mean`；`order=2` 时对残差
    `x - 已累计值` 再做一遍同样的统计并累加。掩码覆盖不到的列输出 0。

    1. 列显著度 `count[c] = sum_r |W[r, c]|`，按 `(count, 原列号)` 降序稳定排序，
       取前 `up_lim` 列作为候选。
    2. 第一段：对 `i = 1 .. up_lim-1`，令 `mask3` 为候选中前 `i` 列，
       用 `hor(., ~mask3, 2) + hor(., mask3, 2)` 重建，取全矩阵 MSE 最小者
       （严格小于才更新，并列保留更小的 `i`）；最终 `i` 对应的列即 `salient_cols`。
    3. 第二段：用最终 `mask3` 算 `group3 = hor(W, mask3, 2)`；把 `mask3` 列置零后
       对 `|W|` 展平排序，在分位点 `q = 0.10, 0.11, ..., 0.90`（共 81 个，
       线性插值）上逐个搜索切分 `split`：
       `mask2 = (~mask3) & (|W| >= split)`，`mask1 = (~mask3) & ~mask2`；
       重建 `hor(W, mask1, 1) + hor(W, mask2, 1) + group3`，取 MSE 最小者
       （严格小于才更新，并列保留更早的分位点）。
    4. 返回该最优切分下的掩码与重建矩阵。

    返回 dict：salient_cols int64 [k]（升序）；split float32 标量；
    mask1/mask2/mask3 bool [O, C]；reconstructed float32 [O, C]。
    内部用 float64 计算。非法输入（W 非二维浮点/空/非有限、up_lim 非整数或越界）
    抛 ValueError。
    """
    # TODO
    raise NotImplementedError
