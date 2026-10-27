import torch


def solve(arrival, num_tokens, deadline, max_batch, max_wait, max_steps):
    """SLO 感知的每步选批：EDF 为主、等待过久的老化优先为辅。

    arrival:(N,) 到达步;num_tokens:(N,) 需要跑的 decode 步数;
    deadline:(N,) 要求在这一步之前（含）完成;max_batch:每步最多同时跑几个;
    max_wait:连续被跳过多少步后升级为「老化优先」;max_steps:模拟步数上限。

    每一步：
      1) 就绪集 = 已到达且还没跑完的请求；
      2) 连续被跳过步数 wait >= max_wait 的请求整批排到最前面（老化，防饿死）；
      3) 组内都按 (deadline 升序, id 升序) 排序，取前 max_batch 个各跑一步；
      4) 被选中的 wait 归零，就绪但没被选中的 wait 加一；
      5) 跑完的记下完成步。

    返回 (start:(N,)（首次运行步，-1 表示没跑过）、finish:(N,)（-1 表示没跑完）、
          missed:int64 标量（超期或未完成的请求数）、promotions:int64 标量（老化入选的总次数）)。
    """
    # TODO
    raise NotImplementedError
