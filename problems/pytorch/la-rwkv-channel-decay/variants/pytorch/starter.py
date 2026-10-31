import torch


def solve(r, k, v, w, u, state=None):
    """逐行转写 RWKV v6 的 WKV 逐通道递推（官方 wkv6 CUDA kernel 的 CPU 账本版）。

    参数
    ----
    r : [T, C]  receptance（未过 sigmoid，本函数不施加，原样乘回）
    k : [T, C]  key
    v : [T, C]  value
    w : [T, C]  data-dependent 衰减（逐 token 逐通道，v6 的 time_decay 项）
    u : [C]     逐通道 bonus（v6 的 time_faaaa 项）
    state : None 或 dict{'a': [C], 'b': [C], 'p': [C]}
            初始状态；None 时 a=b=0、p=-1e38（kernel 的 -inf 代理）

    返回
    ----
    dict：
        wkv         [T, C]  乘 receptance 之前的输出
        y           [T, C]  r * wkv
        final_state dict{'a','b','p'}   终态（[C] 张量各一）
        p_trace     [T][C]  每个 token 处理完后的 p（嵌套 list，float）

    递推（每个通道独立，全程 float32 计算；t 从 0 到 T-1）：
        # 输出分支：bonus u 与旧指数 pp 比
        p_out = max(pp, u)
        e1 = exp(pp - p_out)
        e2 = exp(u - p_out)
        wkv = (e1*aa + e2*v) / (e1*bb + e2)
        # 更新分支：衰减加在旧指数 pp 上，再与当前 k 比
        ww = pp + w
        p = max(ww, k)
        e1 = exp(ww - p)
        e2 = exp(k - p)
        aa = e1*aa + e2*v
        bb = e1*bb + e2
        pp = p
    初始（state 为 None）：aa = 0, bb = 0, pp = -1e38
    """
    raise NotImplementedError
