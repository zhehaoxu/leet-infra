"""VinePPO（arXiv:2410.01679, McGill-NLP/VinePPO）步级 Monte-Carlo 回报账。

请实现 solve(mode=...) 六个入口（官方 TreeEpisodeUtils +
PathAggregator/PathFilter 账，torch/Ray 剥除、账目全入参）：

    mode="score"      tree + gold + pred_map -> 原地填充 score 的树
                      （叶判定 "answer" in node；pred 从 node["answer"]
                      经 pred_map 提取，None -> 0，否则 int(pred == gold)；
                      内部节点 = 子 score 均值、无子 0；is_correct_answer
                      与 score 同值）
    mode="advantage"  已填充 score 的树 -> 原地填充 advantage
                      （child.advantage = child.score - parent.score；
                      叶不赋值、root 无 advantage）
    mode="paths"      tree + include_importance_weights /
                      repeat_early_stopped_paths / branch_factor /
                      max_depth -> [{"node_chain": [...]}]
                      （deepcopy；importance_weight = branch_factor **
                      (max_depth - depth)、达 max_depth 的节点写 1；
                      repeat 时浅于 max_depth 的叶路径按 weight 份
                      extend；逐节点 pop("children")；json round-trip）
    mode="aggregate"  variant="none"/"unique"/"top_k_percent"/
                      "max_k_paths" + paths (+ top_k_percent /
                      keep_at_least_one / reduction / k)
                      （unique 用 json.dumps(sort_keys=True) 集合去重；
                      top_k_percent 按各节点 advantage.get("advantage", 0)
                      的 mean/sum 降序稳定排序取 int(len*p)、
                      keep_at_least_one 时 max(num, 1)；max_k_paths 用
                      random.Random(42).sample）
    mode="filter"     variant="none"/"successful"/"non_zero_score"/
                      "non_zero_last_step_score"/"zero_last_step_score"/
                      "non_zero_last_step_advantage" + 单条 path -> bool
                      （successful 看末节点 is_correct_answer、无
                      "answer" -> False；non_zero_score 看 node_chain[1]；
                      其余看末节点）
    mode="advantage_split"
                      node_chain（含根）+ response_text + input_ids +
                      offset_mapping + tokens ->
                      (advantage_char, advantage_token,
                       clean_response_input_ids, clean_advantage_token)
                      （char: ones(len) * -777 起底，响应文本按首节点
                      "\\n"+text+（多节点再加 "\\n"）/ 中间 text+"\\n" /
                      末 text 逐段广播赋值；token: ones(len(input_ids))
                      * -999 起底，token advantage =
                      advantage_char[offset_mapping[i][0]]；
                      assert tokens[0] == "▁"；清理 = 各自 [1:] 剥首
                      "▁"，ids 先 reshape(-1)）

参考 solution.py 实现你的版本。
"""
import copy
import json
import random

import numpy as np

AGGREGATORS = ("none", "unique", "top_k_percent", "max_k_paths")
FILTERS = ("none", "successful", "non_zero_score",
           "non_zero_last_step_score", "zero_last_step_score",
           "non_zero_last_step_advantage")


def solve(mode="score", tree=None, gold=None, pred_map=None,
          include_importance_weights=False, repeat_early_stopped_paths=False,
          branch_factor=None, max_depth=None,
          variant="none", paths=None, top_k_percent=None,
          keep_at_least_one=True, reduction="mean", k=None,
          node_chain=None, response_text=None, input_ids=None,
          offset_mapping=None, tokens=None):
    """统一入口（判题协议）：按 mode 分发到六个官方账段。"""
    # TODO: 实现六个入口
    raise NotImplementedError
