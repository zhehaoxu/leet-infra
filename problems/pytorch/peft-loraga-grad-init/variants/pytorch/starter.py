"""LoRA-GA: Low-Rank Adaptation with Gradient Approximation (arXiv:2407.05000, Outsider565/LoRA-GA).

LoRA-GA aligns the first-step gradient of the low-rank product with the full
fine-tuning gradient: it takes a randomized low-rank SVD of the accumulated
gradient, picks A/B directions from the top-2r singular-vector bank (four
assignment modes), rescales the factors (four scale modes), then ABSORBS the
initial adapter output into the base weight (weight -= eta * B@A), so training
starts exactly at the pretrained model while the first-step update direction
matches full fine-tuning.

Accounts (official peft fork, tuners/lora/layer.py):
- grad_svd: svd_lowrank(grad.float(), q=min(4r, min(shape)), niter=4); Vt = V.T
- direction: ArBr interleaved (B = U[:, 0:2r:2], A = Vt[1:2r:2, :]); A2rBr
  (B = U[:, :r], A = Vt[r:2r, :]); ArB2r (B = U[:, r:2r], A = Vt[:r, :]);
  random = python random.sample(range(2r), 2r), indexes_A drawn first
- scale: gd (A/=eta, B/=eta), unit (untouched), stable (m**0.25/sqrt(gamma),
  m = out_features), weightS (S(w)/=eta, avg_s = sqrt(S[:r]).mean())
- offset: offset = B@A*eta; weight -= offset; norm_clip keeps the invariant
  (B*sqrt(ratio))@(A*sqrt(ratio))*eta == offset*ratio when ratio < 1
"""
import math
import random

import torch


def grad_svd(grad, r, niter=4, seed=42):
    raise NotImplementedError


def pick_direction(U, Vt, r, direction, seed=0):
    raise NotImplementedError


def apply_scale(B, A, scaling, scale, stable_gamma=1.0, weight_singulars=None):
    raise NotImplementedError


def apply_offset(B, A, scaling, weight, norm_clip=False):
    raise NotImplementedError


def solve(mode=None, **kw):
    if mode == "grad_svd":
        return grad_svd(kw["grad"], kw["r"], niter=kw.get("niter", 4), seed=kw.get("seed", 42))
    if mode == "direction":
        return pick_direction(kw["U"], kw["Vt"], kw["r"], kw["direction"], seed=kw.get("seed", 0))
    if mode == "scale":
        return apply_scale(kw["B"], kw["A"], kw["scaling"], kw["scale"],
                           stable_gamma=kw.get("stable_gamma", 1.0),
                           weight_singulars=kw.get("weight_singulars"))
    if mode == "offset_apply":
        return apply_offset(kw["B"], kw["A"], kw["scaling"], kw["weight"],
                            norm_clip=kw.get("norm_clip", False))
    raise ValueError(f"unknown mode {mode!r}")
