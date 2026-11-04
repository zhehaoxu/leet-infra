"""TaylorSeer 缓存步: 门控 + 差分递推 + 泰勒外推.

回源: diffusers ``hooks/taylorseer_cache.py`` (348 行, 论文 arXiv 2503.06923)。
TaylorSeer 让扩散 Transformer 的模块在部分去噪步跳过前向、用泰勒外推近似输出:

1. 门控 (_measure_should_compute):
       should_compute = (step < disable_cache_before_step)                     # warmup
                     or ((step - disable_cache_before_step - 1) % cache_interval == 0)
                     or (disable_cache_after_step is not None and step >= disable_cache_after_step)
   compute 步执行原始前向并更新泰勒因子; 其余步用 predict 外推。
   注意 warmup 步也要 update (官方注释: factors are updated, caching not applied);
   warmup 结束后的第一个 compute 步是 step = disable_before + 1 (当 (step−before−1)
   首次为 0 的倍数), 之前是 predict 步。

2. 差分递推 (update): 记 last_update_step = L, delta = step − L (>= 1):
       f[0] = output
       f[j+1] = (f[j]_new − f[j]_prev) / delta        # j = 0..max_order−1
   首次更新 (L is None) 只存 f[0]; 若 f[j]_prev 缺失或形状变化则 break (重启展开,
   只保留 order 0)。delta == 0 时官方 raise ValueError。

3. 泰勒外推 (predict): offset = step − L
       output = sum_{j} (offset^j / j!) * f[j]        # j 遍历已有阶数

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 内部递增 state.current_step; 判题域 step 为显式入参, state 不含
  current_step。
* 官方支持 tuple 多输出; 判题域单张量输出 (tuple 扩展语义相同)。
* 官方 taylor_factors_dtype 默认 bfloat16; 判题域固定 float64 (dtype 参数省略)。
* 官方 state 由 StateManager 原地维护; 判题域为纯函数式: 返回新 state dict,
  不修改入参。
* cache_interval 必须 >= 1 (官方 0 会 ZeroDivisionError)、max_order 必须 >= 1
  (官方 0 退化为常数外推), 均为判题域显式校验。
* 官方 max_order=0 / disable_cache_before_step=0 的组合会让首个 step 落入
  predict 且 factors 为空 -> 官方 raise "Cannot predict without prior
  initialization"; 判题域保留该分支 (显式校验)。
"""

import math

import torch


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    extra = set(state) - {"taylor_factors", "last_update_step"}
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    factors = state.get("taylor_factors")
    if not isinstance(factors, dict):
        raise ValueError("state['taylor_factors'] 必须是 dict")
    for k, v in factors.items():
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("taylor_factors 的键必须是非负整数阶数")
        if not isinstance(v, torch.Tensor):
            raise ValueError("taylor_factors 的值必须是 torch 张量")
    lus = state.get("last_update_step")
    if lus is not None and (isinstance(lus, bool) or not isinstance(lus, int) or lus < 0):
        raise ValueError("state['last_update_step'] 必须是非负整数或 None")


def solve(outputs, state, step, cache_interval=5,
          disable_cache_before_step=3, disable_cache_after_step=None, max_order=1):
    """TaylorSeer 缓存步。

    参数
    ----
    outputs : torch.Tensor
        compute 步的原始前向输出 (predict 步忽略此参数, 但类型仍校验)。
    state : dict or None
        缓存状态 {"taylor_factors": {order: tensor}, "last_update_step": int|None};
        首步传 None。
    step : int
        当前去噪步索引 (>= 0)。
    cache_interval : int
        compute 周期 (>= 1)。
    disable_cache_before_step : int
        warmup 步数 (step < 该值的步强制 compute 且只更新因子)。
    disable_cache_after_step : int or None
        cooldown 起点 (step >= 该值的步强制 compute)。
    max_order : int
        泰勒展开最高阶 (>= 1)。

    返回
    ----
    dict(output, computed, state)
        * output   : compute 步为 outputs 原样; predict 步为泰勒外推结果
        * computed : bool, 是否执行了原始前向
        * state    : 新的缓存状态 dict (不修改入参)
    """
    if not isinstance(outputs, torch.Tensor):
        raise ValueError("outputs 必须是 torch 张量")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step 必须是非负整数")
    if isinstance(cache_interval, bool) or not isinstance(cache_interval, int) or cache_interval < 1:
        raise ValueError("cache_interval 必须 >= 1")
    if isinstance(disable_cache_before_step, bool) or not isinstance(disable_cache_before_step, int) \
            or disable_cache_before_step < 0:
        raise ValueError("disable_cache_before_step 必须是非负整数")
    if disable_cache_after_step is not None and (
            isinstance(disable_cache_after_step, bool)
            or not isinstance(disable_cache_after_step, int)
            or disable_cache_after_step < 0):
        raise ValueError("disable_cache_after_step 必须是非负整数或 None")
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
        raise ValueError("max_order 必须 >= 1")
    if state is None:
        state = {"taylor_factors": {}, "last_update_step": None}
    _validate_state(state)

    is_warmup = step < disable_cache_before_step
    is_interval = (step - disable_cache_before_step - 1) % cache_interval == 0
    is_cooldown = disable_cache_after_step is not None and step >= disable_cache_after_step
    should_compute = is_warmup or is_interval or is_cooldown

    factors = state["taylor_factors"]
    last_update_step = state["last_update_step"]

    if should_compute:
        if last_update_step is not None and step == last_update_step:
            raise ValueError("Delta step cannot be zero for TaylorSeer update.")
        new_factors = {0: outputs}
        if last_update_step is not None:
            delta = step - last_update_step
            for j in range(max_order):
                prev = factors.get(j)
                if prev is None or prev.shape != new_factors[j].shape:
                    break
                new_factors[j + 1] = (new_factors[j] - prev.to(outputs.dtype)) / delta
        new_state = {
            "taylor_factors": {o: t.clone() for o, t in new_factors.items()},
            "last_update_step": step,
        }
        return {"output": outputs, "computed": True, "state": new_state}

    # predict 步
    if last_update_step is None or not factors:
        raise ValueError("Cannot predict without prior initialization/update.")
    offset = step - last_update_step
    output = None
    for order in sorted(factors):
        term = factors[order].to(outputs.dtype) * (offset ** order) / math.factorial(order)
        output = term if output is None else output + term
    new_state = {
        "taylor_factors": {o: t.clone() for o, t in factors.items()},
        "last_update_step": last_update_step,
    }
    return {"output": output, "computed": False, "state": new_state}
