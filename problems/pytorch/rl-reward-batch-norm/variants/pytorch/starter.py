import torch


def solve(reward_batches, group_ids, stat_source, momentum, whiten_std, eps):
    """奖励归一化管道：EMA 账本 + 统计源选择 + 白化开关 + 尺度失配探针（纯记账，不推公式）。

    入参：
      reward_batches float (T,B)：按到达顺序排好的 T 个批，每批 B 条奖励
      group_ids      int64 (B,) ：批内每条奖励的 prompt 组号（各批复用同一组号）
      stat_source    0=批全局统计 / 1=running 账本（用本批更新**之前**的值）/ 2=批内组统计
      momentum       float ∈[0,1]，EMA 系数：r ← (1-m)·r + m·批统计（PyTorch BatchNorm 口径）
      whiten_std     1=减均值再除 std，0=只减均值
      eps            float > 0，加在方差上再开方
    返回四元组（float32）：
      normalized    (T,B) 归一化后的奖励
      out_rms       (T,)  每批 normalized 的均方根 = 尺度失配探针（正常应 ≈1）
      running_mean  (T+1,) 账本：[0]=0.0 初值，[t+1]= 第 t 批更新后的 EMA 均值
      running_var   (T+1,) 账本：[0]=1.0 初值，[t+1]= 第 t 批更新后的 EMA 方差
    方差一律总体口径 ddof=0；std = sqrt(所选方差 + eps)；账本永远用批全局统计更新，
    与 stat_source 无关。"""
    # TODO
    raise NotImplementedError
