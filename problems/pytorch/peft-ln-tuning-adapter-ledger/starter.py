"""LN-Tuning 适配器生命周期账（starter）。

实现 solve(mode, **kw)，通过全部用例。模式清单：
  select          目标模块选择（exclude / modules_to_save / target 匹配）
  default_targets 架构默认目标表（未知架构 ValueError）
  update          update_layer 的 deepcopy 整层 + set_adapter 冻结
  trainable       _mark_only_adapters_as_trainable 的可训练参数账
  forward         LNTuningLayer.forward 五分支路由
  merge_unmerge   merge / unmerge 的「交换」状态机
  unload          _unloading_checks + key_list 过滤 + 模块换回

各 mode 的参数语义见题面「模式与规则」，常量清单见 solution 同名常量。
判题环境只有 torch + numpy；本题为纯 Python 控制面，不跑真实模型、不加载 transformers 权重。
"""

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def solve(mode: str, **kw):
    raise NotImplementedError
