import torch


def solve(gen_lens, arrivals, capacity, step_time, batch_size, max_staleness):
    """异步 RL rollout 服务的时间线账：FCFS 准入 + partial rollout 截止 + 按完成顺序成批 + 陈旧度。

    入参（N=请求数，全部整数；不含任何随机量，也不需要生成模型）：
      gen_lens      int64 (N,) 每条请求需要的生成时长（时间单位，>=1）
      arrivals      int64 (N,) 每条请求的到达时刻（不要求有序）
      capacity      int C      生成服务的并发槽数（>=1，槽之间等价）
      step_time     int        训练器一个策略版本的时长（>=1）：版本 v 在时刻 v*step_time 发布，
                               在半开区间 [v*step_time, (v+1)*step_time) 内生效；训练器没数据也
                               照常推进网格（空转），这正是 stall 要量化的东西
      batch_size    int B      每个训练步消费的样本数（>=1，最后一批可以不满）
      max_staleness int S      partial rollout 允许多生的版本数（>=1）
    规则：
      1) 排队顺序 = (arrivals, 请求号) 字典序；请求 i 的开始时刻 start_i 是它轮到「最早空出的槽」
         的时刻，即 start_i = max(arrivals_i, 该槽的空闲时刻)，槽的空闲时刻初值为 0。
      2) 入场版本 v0_i = start_i // step_time；截止线 deadline_i = (v0_i + S) * step_time；
         finish_i = min(start_i + gen_lens_i, deadline_i)；
         truncated_i = 1 当且仅当 finish_i < start_i + gen_lens_i（刚好卡在截止线上完成不算截断）。
      3) 消费顺序 = (finish, 请求号) 字典序；名次 q 的请求进 batch_id = q // B 这一批。
         该批的可用时刻 ready = 批内最晚的 finish，消费版本 c = ready // step_time（整批共用 c）。
      4) mu_i = c_{batch_id_i} - v0_i（这条样本被用掉时已经落后当前策略多少个版本）。
      5) occupancy[tau] = #{i : start_i <= tau < finish_i}，长度 = max_i finish_i（tau 从 0 起）。
      6) stall[k] = max(0, (c_k - k) * step_time)，长度 ceil(N/B)：第 k 批本该在版本 k 被用掉。
    返回五元组（全 int64）：mu (N,)、batch_id (N,)、truncated (N,)、
      occupancy (max finish,)、stall (ceil(N/B),)。N=0 时五个都是长度 0 的 int64 空张量。"""
    # TODO
    raise NotImplementedError
