import torch


def solve(params, lr=1e-2, betas=(0.9, 0.999), eps1=1e-30, eps2=1e-3, decay_rate=-0.8,
          clip_threshold=1.0, scale_parameter=True, weight_decay=0.0):
    # TODO: 返回一个继承 torch.optim.Optimizer 的 Adafactor 实例，step() 就地更新 params。
    #   组步计数 t 从 1 起；β2t = min(betas[1], 1 - t**decay_rate)；b = 1 - β2t
    #   lr_eff = max(eps2, RMS(p))*lr  (scale_parameter=True) 否则 lr
    #   U = g² + eps1
    #   维数≥2：v_r←(1-b)v_r + b*rowmean(U); v_c←(1-b)v_c + b*colmean(U); v̂_ij = v_r_i/sum(v_r) * v_c_j
    #   维数<2：v←(1-b)v + b*U; v̂=v
    #   u = g/√v̂ ; r=RMS(u); factor=max(r/clip_threshold,1); u=u/factor*lr_eff
    #   m←β1*m+(1−β1)*u ; 若 wd≠0: p←p*(1-lr_eff*wd) ; p←p-m
    # 全程 float32；grad 为 None 跳过；禁止调用现成 Adafactor。
    raise NotImplementedError
