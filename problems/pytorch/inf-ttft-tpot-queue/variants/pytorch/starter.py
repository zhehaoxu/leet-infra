def solve(arrivals, prompt_lens, gen_lens, chunk, max_batch):
    """连续批处理的时间线：每一步只能做一件事——推进一个 prefill，或让在跑的请求各出 1 个 token。

    arrivals    : list[int]，第 i 个请求在 arrivals[i] 步到达（允许乱序，互不相同或相同都可以）
    prompt_lens : list[int]，第 i 个请求的 prompt 长度（>=1），prefill 需要处理这么多 token
    gen_lens    : list[int]，第 i 个请求要生成多少个 token（>=1）
    chunk       : int，每次 prefill 步最多推进的 prompt token 数（>=1）
    max_batch   : int，最多同时有多少个活跃请求（>=1）

    时钟以整数步计，第 0 步是第一个时间点。记「活跃请求」= 已准入且尚未完成的请求
    （prefill 中和 decode 中都各占一个 max_batch 名额）。

    每一步的流程：
      1) 准入：只要活跃请求数 < max_batch，就按 (arrivals, 下标) 升序把已到达的等待请求
         逐个放进活跃集，直到满员或没有已到达的请求。本步新准入的请求可以立刻参与本步动作。
      2) 选动作（一步只做一种）：
         a. 活跃集里还有 prompt 没处理完的请求：取「最早准入」（即 FCFS）的那个，
            推进 min(chunk, 剩余) 个 prompt token。本步不做 decode，所有在跑请求都不出词。
         b. 否则若活跃集里有已 prefill 完的请求：做一次 decode，所有已 prefill 完且未完成的
            活跃请求各产出 1 个 token。恰好产满 gen_lens[i] 个的请求在本步完成并让出名额；
            本步让出的名额要等下一步准入才能使用。
         c. 以上都没有：时钟直接跳到下一个到达步，空转不计入步数统计。
      prefill 完成的那一步不产出 token；第一个 token 出现在其后的第一次 decode 步。

    返回 (ttft, decode_span, finish, prefill_steps, decode_steps)：
      ttft[i]        : list[int]，首个 token 的步 - arrivals[i]
      decode_span[i] : list[int]，末个 token 的步 - 首个 token 的步（gen_lens[i]==1 时为 0）
      finish[i]      : list[int]，末个 token 的步
      prefill_steps  : int，全过程中做 prefill 步的次数
      decode_steps   : int，全过程中做 decode 步的次数
    """
    # TODO
    raise NotImplementedError
