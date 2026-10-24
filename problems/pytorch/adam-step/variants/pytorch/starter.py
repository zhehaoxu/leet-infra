import torch


def solve(params, grads, exp_avg, exp_avg_sq, step, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # TODO: 复刻 torch.optim.Adam（amsgrad=False, 无 weight_decay）单步。
    #   m = beta1*exp_avg + (1-beta1)*g;  v = beta2*exp_avg_sq + (1-beta2)*g*g;
    #   bc1 = 1-beta1**step;  bc2 = 1-beta2**step;  step_size = lr/bc1;
    #   denom = v.sqrt()/sqrt(bc2) + eps;  new_params = params - step_size*m/denom。
    #   返回 (new_params, m, v)，不要原地改入参。step 为 1-based。
    raise NotImplementedError
