def solve(pairs, ignore_space=True, is_filter=False):
    """识别指标账（PaddleOCR RecMetric 口径）。

    输入
    ----
    pairs        : [[pred, target], ...] 字符串行对（可空列表）
    ignore_space : 行级预处理——双方先去掉所有空格（官方默认 True）
    is_filter    : 追加归一化——只保留数字与英文字母再转小写（官方默认 False）

    记账（题面钉死，与官方一致）：
      1. 行级预处理顺序：先 ignore_space 双方 `replace(" ", "")`，再 is_filter
         归一化（filter 保留 digits+letters 后 lower）；
      2. dist = 经典 Levenshtein 编辑距离（插入/删除/替换权重均为 (1,1,1)，
         **不含** Damerau 相邻换位）；
      3. ned_i = dist / max(len(pred), len(target))（分母是较长串，
         **不是** len(target)）；两串皆空时 maximum=0，ned_i=0；
      4. exact = 预处理后的 pred == target（严格相等）；
      5. 汇总：acc = exact 行数 / (n + eps)，
         norm_edit_dis = 1 - sum(ned_i) / (n + eps)，eps = 1e-5。
         空批（n=0）照官方公式直接计算：acc = 0.0、norm_edit_dis = 1.0。

    返回
    ----
    dict：
      rows           list[dict]  每行 {pred, target, dist, ned, exact}（pred/target 为预处理后）
      acc            float       严格相等行占比（含 eps 分母）
      norm_edit_dis  float       1 - 逐行 ned 均值（含 eps 分母）
    """
    raise NotImplementedError
