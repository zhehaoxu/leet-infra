import torch


def solve(episode_id, terminated, truncated, actions, rewards, values, final_value):
    """多 episode 轨迹拼装：把多段混合的逐步 rollout 记录重建为训练批次
    (actions, rewards, values, dones, next_values, lengths, starts)。

    入参（全部一维、长度对齐、CPU、无任何随机性）：
      episode_id  int64   (T,)：每步所属 episode 的 id。同一 episode 的步在输入中恰构成
                    一段极大连续段（可乱序，如 [7,7,7,3,3,8,8,8]），id 全局唯一、不再复现。
      terminated  int64   (T,) 0/1：该步因环境真终止（goal/失败）结束其 episode。
      truncated   int64   (T,) 0/1：该步因时间/最长生成本被截断结束其 episode
                    （Gymnasium TimeLimit.truncated 口径）。
      actions     int64   (T,)：该步动作/token id，仅按训练序透传。
      rewards     float32 (T,)：r_t。
      values      float32 (T,)：V(s_t)，critic 对该步状态的预测。
      final_value float32 (K,)：每条 episode 终局观测的 V(s_end)，按**出现段序号**索引
                    （episode_id 第 e 次新出现的连续段即下标 e，0 起）。真终止段的该项
                    存在但不被使用。K = 连续段数 = distinct id 数。
    输入保证：每段恰在末步带且只带一个 flag（terminated 与 truncated 在该步异或为 1），
    其余步两个 flag 均为 0。

    拼装规则：
      * 训练序 = 各段按 episode_id **升序**重排，段内保持原时间顺序；
      * dones int64 (T,)：段末步=1、其余=0。截断结束也算 done——done 的语义是
        「禁止把拼接数组里的下条记录当后继」；
      * next_values float32 (T,)（训练序）三分规则：
          段内非末步          → 同段下一步的 value；
          段末且 truncated=1 → final_value[该段出现段序号]（截断必须 bootstrap 补回）；
          段末且 terminated=1 → 0.0（真终止绝不 bootstrap）。
        任何位置都不许把「下一条 episode 的首步 value」当成后继值（跨段泄漏）。

    返回七元组（全部对齐训练序）：
      actions int64 (T,)、rewards float32 (T,)、values float32 (T,)、
      dones int64 (T,)、next_values float32 (T,)、
      lengths int64 (K,)：训练序中每条 episode 的步数、
      starts int64 (K,)：训练序中每条 episode 的起点（exclusive 前缀和，starts[0]=0）。
    空输入约定：T=0（此时 K=0、final_value 为空）时返回七个 0 元素张量，dtype 同上表。
    CPU float32；禁止现成 RL 库。"""
    # TODO
    raise NotImplementedError
