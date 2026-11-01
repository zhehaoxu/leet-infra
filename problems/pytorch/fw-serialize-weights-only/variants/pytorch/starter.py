def solve():
    """torch.save/weights_only load 白名单账 + load_state_dict 键级账。

    手写判定层：torch.save 后从 zip 包抽 archive/data.pkl，
    pickletools 扫 GLOBAL 指令（protocol 2 全为两行文本 GLOBAL），
    对照题面钉死的允许键表判定 ok/rejected；lambda 在 save 阶段
    PicklingError -> save_fail。不得调用 torch.load /
    Module.load_state_dict。

    官方口径：
      - torch.load 自 2.6 起 weights_only 默认 True，白名单外的 GLOBAL
        -> UnpicklingError；纯 primitive 容器无 GLOBAL 天然放行
      - 白名单（protocol-2 键）：collections OrderedDict/Counter、
        torch Size/device/float32、torch *_Storage、
        torch._utils _rebuild_tensor_v2/_rebuild_parameter、
        __builtin__ set/bytearray/complex、_codecs encode
      - 典型拒绝：torch.nn.modules.linear Linear、types SimpleNamespace、
        __builtin__ xrange（range 对象）、numpy 系
      - load_state_dict：missing/unexpected 各自 sorted；
        strict=True 任一非空 -> RuntimeError（"Error(s) in loading
        state_dict"）；全匹配 -> "All keys matched..."

    返回 dict：allowed_globals / verdicts / sd_full / sd_foreign / sd_extra
    """
    raise NotImplementedError
