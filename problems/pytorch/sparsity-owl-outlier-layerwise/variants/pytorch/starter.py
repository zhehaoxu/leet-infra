import torch


def solve(weights, activation_ms, sparsity, lamda=0.08, hyper_m=3.0):
    """OWL 离群比驱动的逐层稀疏率分配（2310.05175，luuyin/OWL 口径）。
    weights: 嵌套列表 [层][Linear] -> (out,in) 张量；activation_ms 同结构 -> (in,) 逐列均方。
    全部 float32/float64 同 dtype，层数 >= 2、每层非空。
    每层把本层所有 Linear 的 metric=|W|*sqrt(act_ms) 展平拼接：
    outlier_ratio = 严格大于 mean(metric)*hyper_m 的元素占比 * 100；
    再按 (r-min)/(max-min)*2*lamda 归一、减去均值、加 (1-sparsity) 得到 density；
    layer_sparsity = 1-density，<=0 时置 0.01。逐 Linear 按行稳定升序取最小
    int(in*layer_sparsity) 个置零（同分取小列下标）。
    返回 dict：outlier_ratio/density/layer_sparsity（float64 向量），
    metric/mask/pruned_weight（与输入同结构的嵌套列表，mask True=被剪）。
    各层离群比完全相同、权重非二维等非法输入抛 ValueError。"""
    # TODO
    raise NotImplementedError
