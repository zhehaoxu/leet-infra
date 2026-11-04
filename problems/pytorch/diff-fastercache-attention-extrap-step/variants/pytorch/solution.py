"""FasterCache 注意力外推广播步.

回源: diffusers ``hooks/faster_cache.py`` (654 行, FasterCache 论文
arXiv 2410.19355) 的 ``FasterCacheBlockHook``。每步:

1. **跳过门控** (L407-432):
       within = lo < current_timestep < hi               # 开区间 (严格)
       区间外 -> 不跳; 区间内 -> skip = iteration % block_skip_range != 0
       (官方 `iteration > 0 and ...` 的 >0 是 0%N==0 死代码)
       skip 复核: 仅 is_guidance_distilled 或 state.batch_size != 当前 batch
       时才真的跳——cache 只缓存 conditional 半批, denoiser 同时跑 uncond+cond
       (batch == state.batch_size) 时注意力必须真算, 否则形状对不上。
2. **线性外推** (``_compute_approximated_attention_output`` L392-405):
       cache = [t_2, t_1] (最近两次 conditional 输出);
       t_2/t_1 若是整批 (size(0) == 2·batch) 先切 conditional 半批 [batch:];
       output = t_1 + (t_1 - t_2) * weight
   weight 官方由 attention_weight_callback 提供 (None 时默认 0.5), 判题域
   显式入参。
3. **缓存维护** (L459-478):
       cache_output = output; 非 distilled 且 output.size(0) == state.batch_size
       时取 conditional 半批 chunk(2, dim=0)[1];
       cache = [cache[1], cache_output] (首步 [c, c]); iteration += 1。

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 挂注意力模块、当前时间步由 callback 提供; 判题域显式入参。
* 官方分发层 (``_apply_faster_cache_on_attention_class`` 的 identifier 匹配)
  与 767 PAB 同族, 不入本题; block_skip_range/timestep_skip_range 直接入参。
* 官方 weight 是 callback(module); 判题域显式 weight (默认 0.5 同官方缺省)。
* 官方 cache 支持 tuple 多输出分支; 判题域单张量。
* 官方不校验参数; 判题域补张量/区间/weight/state 结构等显式校验。
* state 纯函数式: 返回新 dict, cache 条目 clone。
"""

import torch


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    required = {"iteration", "batch_size", "cache"}
    extra = set(state) - required
    missing = required - set(state)
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    if missing:
        raise ValueError(f"state 缺少键: {sorted(missing)}")
    iteration = state["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("state['iteration'] 必须是非负整数")
    batch_size = state["batch_size"]
    if batch_size is not None and (isinstance(batch_size, bool)
                                   or not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("state['batch_size'] 必须是正整数或 None")
    cache = state["cache"]
    if cache is not None:
        if not isinstance(cache, list) or len(cache) != 2 \
                or not all(isinstance(t, torch.Tensor) for t in cache):
            raise ValueError("state['cache'] 必须是 [t_2, t_1] 两个张量的 list 或 None")


def solve(attention_output, state, current_timestep, block_skip_range=2,
          timestep_skip_range=(-1, 681), weight=0.5, is_guidance_distilled=False):
    """FasterCache 注意力外推广播步。

    参数
    ----
    attention_output : torch.Tensor
        假设真实执行时的注意力输出; size(0) 即本次调用的 batch
        (uncond+cond 整批为 2B, denoiser 跳过 uncond 时为条件半批 B)。
    state : dict or None
        {"iteration", "batch_size", "cache"}; cache 是最近两次 conditional
        输出 [t_2, t_1] 或 None; 首步传 None。
    current_timestep : int
        当前调度器时间步。
    block_skip_range : int
        每 N 次调用算 1 次 (iteration % N == 0), 其余外推广播。
    timestep_skip_range : (lo, hi)
        允许广播的时间步开区间。
    weight : float
        外推权重 (官方 attention_weight_callback 的返回值, 缺省 0.5)。
    is_guidance_distilled : bool
        蒸馏模型无 uncond 分支: 缓存不做半批切分, 跳过不需 batch 失配。

    返回
    ----
    dict(output, computed, state)
        * output   : compute 步为 attention_output; broadcast 步为
                     t_1 + (t_1 - t_2) * weight (必要时切 conditional 半批)
        * computed : 是否真实执行注意力
        * state    : 新 state dict (不修改入参)
    """
    if not isinstance(attention_output, torch.Tensor):
        raise ValueError("attention_output 必须是 torch 张量")
    if attention_output.dim() < 1:
        raise ValueError("attention_output 至少是一维张量")
    if isinstance(current_timestep, bool) or not isinstance(current_timestep, int):
        raise ValueError("current_timestep 必须是整数")
    if isinstance(block_skip_range, bool) or not isinstance(block_skip_range, int) \
            or block_skip_range < 1:
        raise ValueError("block_skip_range 必须 >= 1")
    if isinstance(timestep_skip_range, (str, bytes)) \
            or not isinstance(timestep_skip_range, (list, tuple)) \
            or len(timestep_skip_range) != 2 \
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in timestep_skip_range):
        raise ValueError("timestep_skip_range 必须是 (lo, hi) 二元整数组")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
            or not torch.isfinite(torch.tensor(float(weight))):
        raise ValueError("weight 必须是有限数值")
    if not isinstance(is_guidance_distilled, bool):
        raise ValueError("is_guidance_distilled 必须是 bool")
    if state is None:
        state = {"iteration": 0, "batch_size": None, "cache": None}
    _validate_state(state)
    iteration = int(state["iteration"])
    cache = state["cache"]
    if state["batch_size"] is None:
        batch_size = attention_output.size(0)
    else:
        batch_size = int(state["batch_size"])
    cur_batch = attention_output.size(0)

    lo, hi = timestep_skip_range
    is_within_timestep_range = lo < current_timestep < hi
    if not is_within_timestep_range:
        should_skip_attention = False
    else:
        should_skip_attention = iteration % block_skip_range != 0
    if should_skip_attention:
        should_skip_attention = is_guidance_distilled or batch_size != cur_batch

    if should_skip_attention:
        t_2_output, t_output = cache
        if t_2_output.size(0) != cur_batch:
            t_2_output = t_2_output[cur_batch:]
        if t_output.size(0) != cur_batch:
            t_output = t_output[cur_batch:]
        computed = False
        output = t_output + (t_output - t_2_output) * weight
    else:
        computed = True
        output = attention_output

    cache_output = output
    if not is_guidance_distilled and output.size(0) == batch_size:
        cache_output = output.chunk(2, dim=0)[1]
    if cache is None:
        new_cache = [cache_output.detach().clone(), cache_output.detach().clone()]
    else:
        new_cache = [cache[1], cache_output.detach().clone()]

    new_state = {"iteration": iteration + 1, "batch_size": batch_size,
                 "cache": new_cache}
    return {"output": output, "computed": computed, "state": new_state}
