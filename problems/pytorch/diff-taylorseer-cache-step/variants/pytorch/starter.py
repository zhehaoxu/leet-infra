"""TaylorSeer 缓存步 -- 待实现骨架.

回源: diffusers ``hooks/taylorseer_cache.py`` (TaylorSeerState.update/predict +
TaylorSeerCacheHook._measure_should_compute)。

门控:
    should_compute = (step < disable_cache_before_step)                      # warmup
                  or ((step - disable_cache_before_step - 1) % cache_interval == 0)
                  or (disable_cache_after_step is not None and step >= disable_cache_after_step)

compute 步做差分递推 (delta = step - last_update_step):
    f[0] = output;  f[j+1] = (f[j]_new - f[j]_prev) / delta
    首次更新只存 f[0]; f[j]_prev 缺失或形状变化则 break (重启展开); delta==0 raise。

predict 步做泰勒外推 (offset = step - last_update_step):
    output = sum_j (offset^j / j!) * f[j]        # j 遍历已有阶数
    last_update_step 为 None 或 factors 为空 -> raise。

请实现 solve:
* 校验 outputs/step/cache_interval/before/after/max_order 类型与范围;
* state 为 None (首步) 或 {"taylor_factors": {int: tensor}, "last_update_step": int|None};
* 返回 dict(output, computed, state), state 为新 dict (不修改入参)。
"""

import torch


def solve(outputs, state, step, cache_interval=5,
          disable_cache_before_step=3, disable_cache_after_step=None, max_order=1):
    raise NotImplementedError
