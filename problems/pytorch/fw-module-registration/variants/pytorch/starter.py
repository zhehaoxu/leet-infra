def solve(in_dim, hidden, out_dim):
    """按题面搭建一个混用各种注册方式的模块，返回注册机制账本。

    参数
    ----
    in_dim, hidden, out_dim : 正整数，模块的三个宽度。

    返回
    ----
    dict，键与含义（全部由真实模块产生，不得手填常量）：
        param_names_dedup    list   named_parameters() 默认（去重）按序产出的名字
        param_names_nodup    list   named_parameters(remove_duplicate=False) 的名字
        param_count_dedup    int    去重后参数个数
        param_count_nodup    int   不去重的参数个数
        shared_same_object   bool   模块的 p_first 与 p_second 是否同一个对象
        dedup_keeps_first    bool   共享参数去重时保留的是先注册的 p_first
        buffer_names         list   named_buffers() 按序产出的名字
        state_dict_keys      list   state_dict() 按序的键
        sd_has_duplicate_key bool   共享参数的两个名字是否都出现在 state_dict
        nonpersistent_in_sd  bool   非持久 buffer 是否出现在 state_dict
        placeholder_in_named bool   register_parameter(name, None) 的名字是否被
                                    named_parameters 产出
        placeholder_in_sd    bool   该名字是否出现在 state_dict
        loose_in_modules     bool   普通 list 装的 Linear 是否被 modules() 收录
        module_count         int    modules() 产出个数（含根模块）
        module_linear_count  int    modules() 中 nn.Linear 的个数
        buffer_follows_to    bool   模块 .to(torch.float64) 后持久 buffer 的
                                    dtype 是否跟着变成 float64
        buffer_reassign_kept bool   对持久 buffer 属性重新赋一个普通 Tensor 后，
                                    它是否仍是 buffer 且仍在 state_dict

    模块结构（按此顺序在 __init__ 里搭建）：
        1. self.lin1 = nn.Linear(in_dim, hidden)
        2. shared = nn.Parameter(torch.zeros(out_dim))
           self.p_first = shared
           self.p_second = shared          # 同一对象、两个名字
        3. self.register_buffer('running', torch.zeros(hidden), persistent=True)
        4. self.register_buffer('scratch', torch.ones(2), persistent=False)
        5. self.register_parameter('placeholder', None)
        6. self.loose = [nn.Linear(hidden, 2)]   # 普通 list
        7. self.blocks = nn.ModuleList([nn.Linear(hidden, out_dim)])
        8. 收集完 named_* 与 state_dict 后，再做 buffer_reassign_kept 的实验：
           self.running = torch.randn(hidden)（普通 Tensor）
        9. 最后 .to(torch.float64) 做 buffer_follows_to 的实验

    校验：三个宽度都必须是正整数，否则抛 ValueError。
    """
    raise NotImplementedError
