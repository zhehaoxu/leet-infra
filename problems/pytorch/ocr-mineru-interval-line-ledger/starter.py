"""MinerU 检测框区间与行聚合账（starter）。

实现 solve(mode, **kw)，通过全部用例。模式清单：
  overlap_gate      y/x 轴重叠比例门限（严格 >，退化框保护）
  interval_ledger   闭区间排序合并 + 遮罩 ±1 离散裁除
  line_grouping     INTERLINE 降级 + 竖排判定 + 特殊类型断行 + 行内排序 + 外接矩形
  det_box_merge     angle 框隔离 + 宽高比 > 4 触发同行横向合并

各 mode 的参数语义见题面「模式与规则」，常量清单见 solution 同名常量。
判题环境只有 torch + numpy；本题为纯 Python 数值/整数账，不跑真实检测模型、不加载图像。
"""

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def solve(mode: str, **kw):
    raise NotImplementedError
