"""参考答案：ReduceLROnPlateau —— 继承 torch.optim.lr_scheduler.LRScheduler 的平台期调度器。

与其它调度器不同，本调度器没有「第 t 步解析式」，而是一台依赖外部指标流的状态机：
每次 step(metrics) 更新 (best, num_bad, cooldown_counter) 三个状态量，只有在
「连续变差次数超过 patience」时才把 lr 乘一次 factor。因此调度器不自建 _t，
而是把指标值原样喂进状态机，语义与原生 ReduceLROnPlateau 逐位一致。

状态机（严格按原生顺序，顺序错一步结果就不同）：
  1. cur = float(metrics)
  2. 若 _is_better(cur, best)：best=cur、num_bad=0；否则 num_bad+=1
  3. 若处于冷却（cooldown_counter>0）：cooldown_counter-=1、num_bad=0（本步的变差被吞掉）
  4. 若 num_bad > patience（严格大于）：执行降 lr、cooldown_counter=cooldown、num_bad=0
降 lr：new = max(old*factor, min_lr)；仅当 old-new > eps 才真正写回（否则视作无效降速）。
_is_better（threshold 用于过滤不显著的改善）：
  min/rel: cur < best*(1-threshold)   min/abs: cur < best-threshold
  max/rel: cur > best*(1+threshold)   max/abs: cur > best+threshold
初值 best = mode_worse（min 为 +inf，max 为 -inf），故首个指标必然被判为「更优」并只更新 best。
"""

import math

import torch


class _Plateau(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, mode, factor, patience, threshold, threshold_mode,
                 cooldown, min_lr, eps):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.optimizer = optimizer
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.factor = factor
        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps
        self.min_lr = min_lr
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad = 0
        self.cooldown_counter = 0

    def _is_better(self, cur, best):
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return cur < best * (1.0 - self.threshold)
            return cur < best - self.threshold
        if self.threshold_mode == "rel":
            return cur > best * (1.0 + self.threshold)
        return cur > best + self.threshold

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _reduce_lr(self):
        for g in self.optimizer.param_groups:
            old_lr = float(g["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                g["lr"] = new_lr

    def step(self, metrics, epoch=None):
        cur = float(metrics)
        if self._is_better(cur, self.best):
            self.best = cur
            self.num_bad = 0
        else:
            self.num_bad += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad = 0

        if self.num_bad > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad = 0


def solve(params, base_lr=0.1, factor=0.5, patience=2, min_lr=0.0, mode="min",
          threshold=1e-4, threshold_mode="rel", cooldown=0, eps=1e-8):
    optimizer = torch.optim.SGD(params, lr=base_lr)
    return _Plateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold,
                    threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps)
