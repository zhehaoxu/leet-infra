import torch


def solve(params, grads, buffers, lr, momentum=0.0, weight_decay=0.0):
    # TODO: 复刻 torch.optim.SGD 单步（dampening=0, nesterov=False）。
    #   g = grads + weight_decay*params;  buffers = momentum*buffers + g (momentum>0);
    #   new_params = params - lr*(buffers if momentum>0 else g)。
    #   返回 (new_params, new_buffers)，不要原地改入参。
    raise NotImplementedError
