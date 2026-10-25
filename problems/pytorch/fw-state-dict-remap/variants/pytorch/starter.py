import torch
import torch.nn as nn


def solve():
    """把旧命名 checkpoint 重映射到新模块结构，用 strict=False 加载并返回全部账目。

    新模型（固定结构，参数初值全 0）::

        class Block(nn.Module):        # 子模块名: blocks.0 / blocks.1
            weight     : (out_f, in_f)
            bias       : (out_f,)
            layernorm  : (in_f,)
            gate       : (out_f,)      # 新增层，旧 checkpoint 没有
        class Embed(nn.Module):        # 子模块名: embed
            weight     : (2, 3)
        NewNet: embed=Embed(2,3); blocks=ModuleList([Block(3,2), Block(2,4)])

    旧 checkpoint（键 -> 形状），值按「枚举顺序连续递增」确定：先 0 起步，
    每个键取一段连续整数 reshape 成对应形状::

        transformer.h.0.w  (2,3)   -> 0..5
        transformer.h.0.b  (2,)    -> 6..7
        transformer.h.0.ln (3,)    -> 8..10
        transformer.h.1.w  (4,2)   -> 11..18
        transformer.h.1.b  (4,)    -> 19..22
        transformer.h.1.ln (2,)    -> 23..24
        embed.w            (3,2)   -> 25..30   # 旧 Conv1D 风格 (out,in) 转置存放
        lm_head.weight     (2,2)   -> 31..34   # 新模型没有 lm_head

    重映射规则::

        transformer.h.{i}.w|b|ln  ->  blocks.{i}.weight|bias|layernorm
        embed.w                   ->  embed.weight   （值需转置成 (2,3)）
        其余键原样保留

    返回 dict（顺序与内容都必须符合约定）::

        {
          "mapping": {旧键: 新键, ...},          # 只含发生了重命名的键，按旧键字典序排序
          "missing_keys": sorted list[str],     # 新模型有、映射后 checkpoint 没有
          "unexpected_keys": sorted list[str],  # checkpoint 有、新模型没有
          "loaded": {参数名: 展平 list[float]},  # 加载后 model.state_dict() 逐参数行主序展平
        }

    提示：`load_state_dict(..., strict=False)` 返回的 `_IncompatibleKeys` 中两个列表
    不保证有序，返回前需自行 `sorted`。
    """
    # TODO: 构造旧 checkpoint、重映射键（embed 需转置值）、strict=False 加载、汇总四类账
    raise NotImplementedError
