import torch


def solve(weight, inputs, activation_ms, target_sparsity, iterations=10,
          qmetric="cosim_flat", qmetric_dimwise="cosim", forbid_neg_lr=False,
          lr_grid=(0.01, 0.02, 0.04, 0.08, 0.12, 0.16),
          neg_lr_grid=(-0.01, -0.02, -0.04)):
    """TRIM 逐行稀疏率迭代（2505.16743，flobk/TRIM 口径）。
    weight:(out,in)、inputs:(in,tokens)、activation_ms:(in,) 同 dtype（float32/float64），out>=2。
    排序键 = |W|*sqrt(activation_ms)；每行按稳定升序取前 int(in*s_row) 个小值（同分取小列下标）。
    从均匀 s_row=target 出发，每轮用「未剪输出 vs 当前剪枝输出」的逐行余弦：
      d=(qm-qm.min())/(qm.max()-qm.min()+1e-6); d*=lr*2; d-=d.mean();
      x=clip(x+d,0,0.95); x=x-mean(x)+target；
    取整段迭代中 flat 余弦最大的 x。正 lr 网格首个不再提升即停；若正 lr 从未超过
    基线且 forbid_neg_lr=False，再试负 lr 网格。最后对最优 x 做 >0.95→0.95、<0→0、
    再减均值加 target 的后处理，生成掩码。只支持 qmetric='cosim_flat'。
    返回 dict：sparsities(float64) / keep_mask(bool) / pruned_weight /
    baseline_metric / best_metric / learning_rate / best_iter / starting_var / ending_var。
    非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
