import torch


def solve(models, model_device, phase_model, phase_device, phase_train,
          resident, phase_time, hbm_cap, num_devices):
    """RLHF 多模型（actor/critic/ref/reward）colocate 的显存 + 时间线记账（整数表算，不做前向）。

    入参（M=模型数，P=阶段数，D=设备数）：
      models       int64 (M,3)：每模型 [param_bytes, opt_bytes, act_bytes]，推理模型 opt 记 0
      model_device int64 (M,)：模型归属设备 id
      phase_model  int64 (P,)：本阶段被使用/被更新的模型 id
      phase_device int64 (P,)：本阶段跑在哪台设备
      phase_train  int64 (P,)：0/1，本阶段是否更新 phase_model（只有 1 时把 phase_model 的 act
                     摊到 phase_device 上；优化器态不靠它，而是随 resident 整块进出）
      resident     int64 (P,M)：1=该模型完整训练态(param+opt)此刻在该模型的设备上，0=已 offload
      phase_time   int64 (P,)：本阶段耗时（任意时间单位）
      hbm_cap      int：单卡显存上限
      num_devices  int：D（输出向量长度）
    返回五元组：
      peak       int64 (D,) 每设备一步内显存峰值 = max_p(Σ 该设备上 resident 模型的 param+opt
                     + (phase_train[p]==1 and phase_device[p]==d) * act[phase_model[p]])
      offload    int64 (D,) 每设备 offload/onload 传输字节 = 翻转次数 ×(param+opt)，
                     时间线以 phase 0 的配置为稳态、步末必须回到该稳态（收尾多算一次翻转）
      idle_time  int64 (D,) 每设备一步内空闲总时长 = Σ_p phase_time[p]·[phase_device[p]!=d]
      max_idle   int64 (D,) 每设备最长的**连续**空闲阶段时长之和（不跨步循环）
      feasible   bool (D,)  peak <= hbm_cap
    P=0 时 peak/offload/idle_time/max_idle 全 0、feasible 全 True。"""
    # TODO
    raise NotImplementedError
