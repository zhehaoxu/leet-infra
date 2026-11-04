"""Pyramid Attention Broadcast 广播步.

回源: diffusers ``hooks/pyramid_attention_broadcast.py`` (314 行, PAB 论文
arXiv 2408.12588) 与 ``hooks/_common.py`` 的 identifier 常量。每步两段:

1. **注意力层分发** (``_apply_pyramid_attention_broadcast_on_attention_class``
   L241-287): 对层名做 `re.search` 任意 identifier 匹配, 三类按优先级:
       spatial  = match(spatial_ids)   and spatial_range  is not None and not is_cross
       temporal = match(temporal_ids)  and temporal_range is not None and not is_cross
       cross    = match(cross_ids)     and cross_range    is not None and is_cross
   spatial > temporal > cross; 全不匹配 -> 不挂 hook (恒 compute, state 不动)。
   注意 `re.search("blocks", "temporal_transformer_blocks.0.attn1")` 命中——
   官方默认 identifier 下 temporal 层名会先被 spatial 认领 (优先级即语义)。
   三个 block_skip_range 全为 None 时官方回退 spatial=2 (带 warning)。
2. **广播门控** (``PyramidAttentionBroadcastHook.new_forward`` L156-174):
       within         = lo < current_timestep < hi        # 开区间 (严格)
       should_compute = cache is None or not within or iteration % block_skip_range == 0
   compute -> output = 真实注意力输出; broadcast -> output = cache (上次输出)。
   state.cache 每次调用都更新为本次 output, iteration 每次调用 +1。
   (官方 `iteration == 0` 分支被 `0 % n == 0` 完全子sumed, 是死代码。)

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 挂在每个注意力模块上、current_timestep 由 callback 提供; 判题域
  单函数显式传层名/是否 cross/当前时间步, 分发与门控合一。
* 官方 `or self.state.iteration == 0` 死代码不入实现 (0 % n == 0 恒真)。
* 官方 broadcast 时 cache 存引用; 判题域存 clone (纯函数式, 不改入参)。
* 官方不校验参数; 判题域补 name/is_cross/张量/state/skip_range >= 1/时间步
  区间二元组等显式校验。
* 未匹配层返回 block_type=None、computed=True、state 原样透传。
"""

import re

import torch

_SPATIAL_IDENTIFIERS_DEFAULT = (
    "blocks",
    "transformer_blocks",
    "single_transformer_blocks",
    "layers",
    "visual_transformer_blocks",
)
_TEMPORAL_IDENTIFIERS_DEFAULT = ("temporal_transformer_blocks",)
_CROSS_IDENTIFIERS_DEFAULT = ("blocks", "transformer_blocks", "layers")


def _validate_identifiers(identifiers, name):
    if isinstance(identifiers, str) or not isinstance(identifiers, (list, tuple)) \
            or not all(isinstance(i, str) for i in identifiers):
        raise ValueError(f"{name} 必须是字符串序列")


def _validate_timestep_range(range_value, name):
    if isinstance(range_value, (str, bytes)) or not isinstance(range_value, (list, tuple)) \
            or len(range_value) != 2 \
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in range_value):
        raise ValueError(f"{name} 必须是 (lo, hi) 二元整数组")


def _dispatch(name, is_cross_attention, spatial_range, temporal_range, cross_range,
              spatial_ids, temporal_ids, cross_ids):
    is_spatial = any(re.search(identifier, name) is not None for identifier in spatial_ids) \
        and spatial_range is not None and not is_cross_attention
    is_temporal = any(re.search(identifier, name) is not None for identifier in temporal_ids) \
        and temporal_range is not None and not is_cross_attention
    is_cross_match = any(re.search(identifier, name) is not None for identifier in cross_ids) \
        and cross_range is not None and is_cross_attention

    if is_spatial:
        return spatial_range, "spatial"
    if is_temporal:
        return temporal_range, "temporal"
    if is_cross_match:
        return cross_range, "cross"
    return None, None


def solve(name, is_cross_attention, hidden_states, state, current_timestep,
          spatial_attention_block_skip_range=None,
          temporal_attention_block_skip_range=None,
          cross_attention_block_skip_range=None,
          spatial_attention_timestep_skip_range=(100, 800),
          temporal_attention_timestep_skip_range=(100, 800),
          cross_attention_timestep_skip_range=(100, 800),
          spatial_attention_block_identifiers=_SPATIAL_IDENTIFIERS_DEFAULT,
          temporal_attention_block_identifiers=_TEMPORAL_IDENTIFIERS_DEFAULT,
          cross_attention_block_identifiers=_CROSS_IDENTIFIERS_DEFAULT):
    """PAB 广播步: 注意力层分发 + 迭代广播门控。

    参数
    ----
    name : str
        注意力层全名 (官方 named_modules 的 key)。
    is_cross_attention : bool
        层的 is_cross_attention 标志。
    hidden_states : torch.Tensor
        假设真实执行时的注意力输出。
    state : dict or None
        {"iteration", "cache"}; 首步传 None。
    current_timestep : int
        当前调度器时间步 (官方 current_timestep_callback() 的值)。
    *_block_skip_range : int or None
        各类注意力的广播间隔 N (每 N 次调用算 1 次, 其余 N-1 次复用缓存);
        三者全 None 时官方回退 spatial=2。
    *_timestep_skip_range : (lo, hi)
        各类注意力的时间步开区间 (lo, hi), 仅区间内才允许 broadcast。
    *_block_identifiers : sequence[str]
        层名匹配用的 re.search 正则片段。

    返回
    ----
    dict(output, computed, block_type, state)
        * output     : compute 步为 hidden_states; broadcast 步为缓存输出;
                       未匹配层恒为 hidden_states
        * computed   : 是否真实执行注意力
        * block_type : "spatial" / "temporal" / "cross" / None (未匹配)
        * state      : 新 state dict (未匹配层原样透传; 不修改入参)
    """
    if not isinstance(name, str):
        raise ValueError("name 必须是字符串")
    if not isinstance(is_cross_attention, bool):
        raise ValueError("is_cross_attention 必须是 bool")
    if not isinstance(hidden_states, torch.Tensor):
        raise ValueError("hidden_states 必须是 torch 张量")
    if isinstance(current_timestep, bool) or not isinstance(current_timestep, int):
        raise ValueError("current_timestep 必须是整数")
    for range_name, range_value in (
        ("spatial_attention_block_skip_range", spatial_attention_block_skip_range),
        ("temporal_attention_block_skip_range", temporal_attention_block_skip_range),
        ("cross_attention_block_skip_range", cross_attention_block_skip_range),
    ):
        if range_value is not None and (isinstance(range_value, bool)
                                        or not isinstance(range_value, int) or range_value < 1):
            raise ValueError(f"{range_name} 必须是 >= 1 的整数或 None")
    for range_name, range_value in (
        ("spatial_attention_timestep_skip_range", spatial_attention_timestep_skip_range),
        ("temporal_attention_timestep_skip_range", temporal_attention_timestep_skip_range),
        ("cross_attention_timestep_skip_range", cross_attention_timestep_skip_range),
    ):
        _validate_timestep_range(range_value, range_name)
    for ids_name, ids_value in (
        ("spatial_attention_block_identifiers", spatial_attention_block_identifiers),
        ("temporal_attention_block_identifiers", temporal_attention_block_identifiers),
        ("cross_attention_block_identifiers", cross_attention_block_identifiers),
    ):
        _validate_identifiers(ids_value, ids_name)
    if state is None:
        state = {"iteration": 0, "cache": None}
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    required = {"iteration", "cache"}
    extra = set(state) - required
    missing = required - set(state)
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    if missing:
        raise ValueError(f"state 缺少键: {sorted(missing)}")
    iteration = state["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("state['iteration'] 必须是非负整数")
    cache = state["cache"]
    if cache is not None and not isinstance(cache, torch.Tensor):
        raise ValueError("state['cache'] 必须是 torch 张量或 None")

    # 三者全 None: 官方回退 spatial_attention_block_skip_range=2
    if spatial_attention_block_skip_range is None \
            and temporal_attention_block_skip_range is None \
            and cross_attention_block_skip_range is None:
        spatial_attention_block_skip_range = 2

    block_skip_range, block_type = _dispatch(
        name, is_cross_attention,
        spatial_attention_block_skip_range, temporal_attention_block_skip_range,
        cross_attention_block_skip_range,
        spatial_attention_block_identifiers, temporal_attention_block_identifiers,
        cross_attention_block_identifiers)

    if block_skip_range is None:
        # 未匹配: 官方不挂 hook, 恒真实执行, state 不动
        return {"output": hidden_states, "computed": True, "block_type": None,
                "state": dict(state)}

    lo, hi = None, None
    if block_type == "spatial":
        lo, hi = spatial_attention_timestep_skip_range
    elif block_type == "temporal":
        lo, hi = temporal_attention_timestep_skip_range
    else:
        lo, hi = cross_attention_timestep_skip_range

    is_within_timestep_range = lo < current_timestep < hi
    should_compute = cache is None or not is_within_timestep_range \
        or iteration % block_skip_range == 0

    if should_compute:
        computed = True
        output = hidden_states
    else:
        computed = False
        output = cache

    new_state = {"iteration": iteration + 1, "cache": output.detach().clone()}
    return {"output": output, "computed": computed, "block_type": block_type,
            "state": new_state}
