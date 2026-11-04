"""First Block Cache (FBCache) 残差门控步.

回源: diffusers ``hooks/first_block_cache.py`` (258 行, ParaAttention 的 First
Block Cache, 基于 TeaCache 思想 arXiv 2411.19108)。Transformer 的 N 个块中:
首块 (head) 永远真实执行, 其残差决定剩余块是否跳过:

    residual = head_output - head_input                     # 首块残差
    prev     = state["head_block_residual"]                 # 上次 compute 的残差
    diff     = absmean(residual - prev) / absmean(prev)     # 相对 L1 变化
    compute  <=>  prev is None or diff > threshold          # 首步强制 compute

* compute 路径: 全部块真实执行, 最终输出 = tail_output (尾块真实输出);
  state 更新 head_block_output = head_output、head_block_residual = residual、
  tail_block_residuals = tail_output - head_block_output (尾块相对首块输出的残差)。
* cache 路径 (diff <= threshold): 中间块透传, 最终输出 = state["tail_block_residuals"]
  + head_output (缓存的尾残差 + 当前首块输出); state 三元组全部保持旧值
  (连续 cache 步都与上次 compute 的残差比较 -- stale 比较)。

与官方实现的判题域约定 (official -> judgement):
* 官方 hook 挂在 head/中间/尾块上共享 StateManager; 判题域把一次前向压缩为
  (head_input, head_output, tail_output) 三个张量 + 显式 state, 中间块的透传
  在 cache 路径下是恒等映射, 数学上等价。
* 官方支持 tuple (hidden_states, encoder_hidden_states) 双输出; 判题域单张量。
* 官方 reset() 只清 tail_block_residuals 与 should_compute, 保留 head_block_output
  / head_block_residual (跨 run 复用残差比较); 判题域 state 由调用方完全掌控,
  三元组均可独立为 None; cache 路径遇到 tail_block_residuals 为 None 时判题域
  显式拒绝 (官方会 TypeError)。
* diff 分母 absmean(prev) == 0 时官方产生 0/0=nan (nan > x 为 False -> cache)
  或 x/0=inf (-> compute); 判题域显式规范化为同语义的 0.0 / inf, 避免浮点警告。
* threshold 必须 >= 0 (官方默认 0.05, 无校验; 负阈值使 cache 永不触发的语义
  无意义, 判题域拒绝)。
"""

import torch


def _absmean(t):
    return float(t.detach().abs().to(torch.float64).mean().item())


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("state 必须是 dict 或 None")
    extra = set(state) - {"head_block_output", "head_block_residual", "tail_block_residuals"}
    if extra:
        raise ValueError(f"state 含未知键: {sorted(extra)}")
    for k in ("head_block_output", "head_block_residual", "tail_block_residuals"):
        v = state.get(k)
        if v is not None and not isinstance(v, torch.Tensor):
            raise ValueError(f"state['{k}'] 必须是 torch 张量或 None")


def solve(head_input, head_output, tail_output, state=None, threshold=0.05):
    """First Block Cache 残差门控步。

    参数
    ----
    head_input : torch.Tensor
        首块输入 (整模型的输入 hidden_states)。
    head_output : torch.Tensor
        首块真实执行 (假设完整前向) 的输出。
    tail_output : torch.Tensor
        尾块真实执行 (假设完整前向) 的输出 (compute 路径的最终输出)。
    state : dict or None
        {"head_block_output", "head_block_residual", "tail_block_residuals"},
        值均为张量或 None; 首步传 None。
    threshold : float
        相对 L1 残差阈值 (>= 0), diff > threshold 时 compute。

    返回
    ----
    dict(final_output, computed, diff, state)
        * final_output : compute 路径为 tail_output; cache 路径为
          tail_block_residuals + head_output
        * computed     : bool
        * diff         : 相对 L1 残差变化 (float; 首步为 None)
        * state        : 新 state dict (不修改入参)
    """
    for name, t in (("head_input", head_input), ("head_output", head_output),
                    ("tail_output", tail_output)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} 必须是 torch 张量")
    if not (head_output.shape == head_input.shape == tail_output.shape):
        raise ValueError("head_input/head_output/tail_output 形状必须一致")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or threshold < 0:
        raise ValueError("threshold 必须是非负数值")
    if state is None:
        state = {"head_block_output": None, "head_block_residual": None,
                 "tail_block_residuals": None}
    _validate_state(state)

    prev_residual = state["head_block_residual"]
    residual = head_output - head_input

    if prev_residual is None:
        diff = None
        computed = True
    else:
        num = _absmean(residual - prev_residual)
        den = _absmean(prev_residual)
        if den == 0.0:
            diff = 0.0 if num == 0.0 else float("inf")
        else:
            diff = num / den
        computed = diff > threshold

    if computed:
        new_state = {
            "head_block_output": head_output.clone(),
            "head_block_residual": residual.clone(),
            "tail_block_residuals": (tail_output - head_output).clone(),
        }
        final = tail_output
    else:
        cached_tail = state["tail_block_residuals"]
        if cached_tail is None:
            raise ValueError("cache 路径需要 state['tail_block_residuals'] (官方会 TypeError)")
        new_state = {k: (v.clone() if v is not None else None) for k, v in state.items()}
        final = cached_tail + head_output

    return {"final_output": final, "computed": computed, "diff": diff, "state": new_state}
