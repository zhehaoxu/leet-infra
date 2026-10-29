def solve(arrivals, service, priority, max_batch, aging_interval, priority_cap):
    """带优先级与老化（aging）的准入调度：低优请求等得够久就逐档提升有效优先级。

    arrivals       : list[int]，第 i 个请求在第 arrivals[i] 步到达（>=0，允许重复与乱序）
    service        : list[int]，第 i 个请求一旦准入需要连续占用 service[i] 步（>=1）
    priority       : list[int]，第 i 个请求的基础优先级（>=0，数值越大越重要）
    max_batch      : int，最多同时有几个请求在跑（>=1）
    aging_interval : int，每等待 aging_interval 步，有效优先级涨 1 点（>=1）
    priority_cap   : int，有效优先级的上限（>= max(priority)）

    时钟以整数步计，第 0 步是第一个时刻。每个 step：
      1) 准入：空闲名额 = max_batch − 正在跑的请求数。在所有「已到达且未准入」的请求里
         逐个挑选，每次挑「当前时刻 t 下」有效优先级最高者：
             有效优先级 = min(priority_cap, priority[i] + (t − arrivals[i]) // aging_interval)
         打平时依次比：基础优先级高者优先 → 到达更早者优先 → 下标更小者优先。
         本步选中的请求从本步开始占用名额并执行服务。
      2) 执行：所有在跑的请求各消耗 1 步服务；服务在该步结束的请求于本步完成（完成步 = 本步），
         它让出的名额从下一步起可用。
      3) 空转：若没有请求在跑且也没有已到达的请求，把时钟直接推到下一个到达步，空转不计任何账。

    返回 (admit, finish, max_wait)：
      admit    : list[int]，第 i 个请求的准入步（绝对步号）
      finish   : list[int]，第 i 个请求的完成步（= admit[i] + service[i] − 1）
      max_wait : int，所有请求里最大的「准入步 − 到达步」
    """
    # TODO
    raise NotImplementedError
