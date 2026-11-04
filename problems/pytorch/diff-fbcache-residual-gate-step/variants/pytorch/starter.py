"""First Block Cache (FBCache) 残差门控步 -- 待实现骨架.

回源: diffusers ``hooks/first_block_cache.py`` (FBCHeadBlockHook /
_should_compute_remaining_blocks)。Transformer 的首块永远真实执行, 其残差
决定剩余块是否跳过:

    residual = head_output - head_input
    diff     = absmean(residual - prev_residual) / absmean(prev_residual)
    compute  <=>  prev_residual is None or diff > threshold

* compute: 最终输出 = tail_output; state 更新 head_block_output = head_output、
  head_block_residual = residual、tail_block_residuals = tail_output - head_output。
* cache (diff <= threshold): 最终输出 = tail_block_residuals + head_output;
  state 保持旧值 (连续 cache 步与上次 compute 的残差做 stale 比较)。
* 分母 absmean(prev) == 0: 规范化为 diff=0.0 (分子也为 0 -> cache) 或 inf -> compute。

请实现 solve:
* 校验 head_input/head_output/tail_output 类型与形状一致; threshold 非负数值;
  state 结构 (三键白名单, 值为张量或 None);
* 返回 dict(final_output, computed, diff, state), state 为新 dict (不修改入参);
  cache 路径 tail_block_residuals 为 None 时 raise。
"""

import torch


def solve(head_input, head_output, tail_output, state=None, threshold=0.05):
    raise NotImplementedError
