import torch


def solve(keys, values, scores, conv_weight, normalizer, hh_keep_rate, local_len, mem_size):
    """LoCoCo 卷积重算合并（2406.05317，H2O 硬保留 + token 轴卷积软合并）。
    keys/values:(S,H,D) NHD；scores:(H,S) 累计分数；conv_weight:(m,2D,K)，K 为奇数；
    m=int(mem_size*(1-hh_keep_rate))，keep=mem_size-m，n=S-keep；normalizer 为标量。
    hh_idx:(H,keep) 升序（末 local_len 个恒留，且不参与前缀 topk）；
    res_idx:(H,m) 升序的非重击代表；conv_prob:(H,m,n)=softmax(relu(conv([k;v])))；
    blend:(H,m,n)=one_hot(res)*(1-normalizer)+conv_prob*normalizer；
    merged_keys/values:(H,mem_size,D)=[重击 token; blend@非重击 K/V]。
    内部 float64，返回输入 dtype；非法输入抛 ValueError。返回 dict 6 键。"""
    # TODO
    raise NotImplementedError
