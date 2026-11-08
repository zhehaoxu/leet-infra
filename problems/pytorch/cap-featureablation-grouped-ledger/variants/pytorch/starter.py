#!/usr/bin/env python3
"""856 骨架：Captum FeatureAblation 分组消融账。

实现五个 mode，全部通过 solve(mode, **kw) 分派：
  mask_range  —— 默认/显式 feature_mask、max+1 与 max-min+1 双口径、ppe 分批
  ablate      —— 堆叠 0/1 掩码 + baseline 混合（_construct_ablated_input）
  schedule    —— ppe 重复、尾批重展开、target/additional args 展开、kwargs 选择
  eval_out    —— 初始 eval 展平、差值累加、use_weights 权重除法
  attribution —— 六种确定性 forward_kind 与整条 attribute 编排

口径提示：
  - 纯 Python，不依赖 torch / captum / numpy，不依赖全局 RNG。
  - 输入是嵌套列表；输出是题目约定的复合字典，字段必须与机制口径一致。
  - 所有非法输入抛 ValueError，消息点名被违反的约束。
  - 注意 num_features = max+1，而 feature_count = max-min+1，两者不可混用。
"""

import math


# ----------------------------------------------------------------- 工具（可自行补充）
def shape_of(x):
    """返回嵌套列表的形状；标量返回 []。"""
    raise NotImplementedError


def flatten(x):
    """按 C 序展平成一层。"""
    raise NotImplementedError


def unflatten(vals, shape):
    """把展平数据按形状折回嵌套列表。"""
    raise NotImplementedError


def numel(shape):
    """形状对应的元素个数。"""
    raise NotImplementedError


def broadcast_shapes(a, b):
    """NumPy 风格广播：尾维对齐，每维相等或其中之一为 1。"""
    raise NotImplementedError


def broadcast_to(x, shape):
    """把嵌套列表广播到目标形状（标量需单独处理）。"""
    raise NotImplementedError


def bop(a, b, op):
    """广播后的逐元素二元运算。"""
    raise NotImplementedError


def sum_axis0(x):
    """沿第 0 维求和，返回 shape[1:] 形状的结果。"""
    raise NotImplementedError


def map_vals(x, fn):
    """对每个标量套用 fn，保持嵌套结构。"""
    raise NotImplementedError


def repeat_rows(x, k):
    """整份输入按行重复 k 次（不是逐元素 repeat）。"""
    raise NotImplementedError


# ----------------------------------------------------------------- 输入校验
def _ppe(kw):
    """ perturbations_per_eval 必须是 >=1 的整数；官方 bool 视为 int。"""
    raise NotImplementedError


def _validate_mask(mask, ishape):
    """rank / 首维 / 其余维 / 非负整数值。"""
    raise NotImplementedError


def _fmt_masks(inputs, feature_mask):
    """None 时生成默认 mask；多输入 offset 全局累加。"""
    raise NotImplementedError


def _fmt_baselines(inputs, baselines):
    """None 时为标量 0；校验数量与首维。"""
    raise NotImplementedError


def _mask_eq(mask, j):
    """(mask == j) 的 0/1 掩码。"""
    raise NotImplementedError


# ----------------------------------------------------------- mode 1
def mask_range(kw):
    """返回 mask / mask_shape / numel_per_example / min_feature /
    num_features / feature_count / loop_start / num_steps / ppe_clamped /
    chunks / num_chunks / progress_total。"""
    raise NotImplementedError


# ----------------------------------------------------------- mode 2
def ablate(kw):
    """返回 mask / mask_shape / ablated / ablated_shape /
    num_ablated_features / ones_per_feature / baseline_shape。"""
    raise NotImplementedError


# ----------------------------------------------------------- mode 3
def schedule(kw):
    """返回 meta 字段 + yields（每个 yield 含 start/end/k/batch_size/
    current_inputs/current_mask/current_target/current_additional_args）。"""
    raise NotImplementedError


# ----------------------------------------------------------- mode 4
def eval_out(kw):
    """返回 attributions / weights / flattened_initial_eval / attrib_type /
    n_outputs / total_attrib_shapes / eval_diffs / initial_shape。"""
    raise NotImplementedError


# ----------------------------------------------------------- mode 5
def attribution(kw):
    """返回 attributions / weights / n_outputs / attrib_type /
    flattened_initial_eval / initial_shape / num_forwards / forward_shapes /
    chunks / progress_total。"""
    raise NotImplementedError


def solve(mode, **kw):
    if mode == "mask_range":
        return mask_range(kw)
    if mode == "ablate":
        return ablate(kw)
    if mode == "schedule":
        return schedule(kw)
    if mode == "eval_out":
        return eval_out(kw)
    if mode == "attribution":
        return attribution(kw)
    raise ValueError("unknown mode: %r" % (mode,))
