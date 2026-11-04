"""VinePPO（arXiv:2410.01679, McGill-NLP/VinePPO）步级 Monte-Carlo 回报账。

VinePPO 的核心主张：LLM 推理题上价值网难训（每一个中间步都要求"从这里走
下去最终对"的期望），不如**不要价值网**——在每步分支处续采 k 条完整轨迹、
用答案正确性当回报，把每步的 MC 回报均值当 value、advantage =
child.score − parent.score 的**纯树记账**。官方 TreeEpisodeUtils +
PathAggregator/PathFilter 全注册类（torch/Ray/多进程剥除，账目全入参），
本题逐段复刻：

solve(mode=...) 六个入口（官方逐条钉死）：

    mode="score"（TreeEpisodeUtils.compute_score）：
        tree + gold + pred_map（answer 文本 -> 提取的预测答案或缺失）->
        原地填充 score 后返回同一棵树。叶子判定是 **"answer" in node**
        （不是"没有 children"）；预测答案从 **node["answer"] 经 pred_map
        提取——node 里自带的 "pred" 字段完全不参与**（官方 extract_
        pred_answer 只看 answer）；提取为 None -> score 0，否则
        int(pred == gold)；is_correct_answer 冗余写同值；内部节点 score =
        子 score 均值（无子 -> 0）。root 的 score = 各子树均值。

    mode="advantage"（compute_advantage）：
        tree（须已填充 score）-> 原地填充 advantage。**child.advantage =
        child.score − parent.score**（树形差分、无自底向上 bias 传播）；
        叶子直接 return 不赋值；root 无 advantage。

    mode="paths"（extract_paths_from_tree）：
        tree + include_importance_weights/repeat_early_stopped_paths/
        branch_factor/max_depth -> [{"node_chain": [root..leaf]}] 列表。
        输入树先 **deepcopy**；importance_weight = branch_factor **
        (max_depth − depth)，但写入节点的是 max_depth > depth 时的该值、
        达到 max_depth 的节点写 **1**；repeat_early_stopped_paths 时浅于
        max_depth 的叶路径按 **importance_weight 份** extend（max_depth
        处的叶只 1 份）；遍历过程**顺手逐节点 pop("children")**（序列化
        提速的副作用）；最后 json round-trip（元组变列表、数值保真）。

    mode="aggregate"（PathAggregator 注册类分发）：
        variant="none" -> 原列表原样返回；
        variant="unique" -> **json.dumps(p, sort_keys=True) 进 set** 去
            重再 json.loads 还原（set 无序，输出顺序不可复现，只能比集合）；
        variant="top_k_percent" -> 构造期 assert 0 < p <= 1；空列表原样
            返回；排序键是每条 path 各节点 **advantage.get("advantage", 0)**
            （根节点无 advantage 记 0）的 mean/sum（reduction），sorted
            **reverse=True 稳定排序**；num_to_keep = int(len * p)，
            keep_at_least_one 时 max(num, 1)；
        variant="max_k_paths" -> len(paths) <= k 原样返回，否则
            **random.Random(42).sample(paths, k)**（种子固定）。

    mode="filter"（PathFilter 注册类，单条 path -> bool）：
        variant="successful" -> 末节点无 "answer" -> False，否则返回末
            节点 **is_correct_answer**（0/1）；
        variant="non_zero_score" -> 看 **node_chain[1]（第一步）** 的
            score > 0——不是末节点；
        variant="non_zero_last_step_score" / "zero_last_step_score" /
        "non_zero_last_step_advantage" -> 末节点 score>0 / score==0 /
            advantage>0。

    mode="advantage_split"（convert_path_to_episode 的字符级分摊段）：
        node_chain（含根）+ response_text + input_ids + offset_mapping +
        tokens -> (advantage_char, advantage_token, clean_response_input_ids,
        clean_advantage_token) 全 list。字符账：ones(len(response_text))
        * **-777** 哨兵起底；响应文本按 **首节点 "\\n"+text+（多节点再加
        "\\n"）/ 中间节点 text+"\\n" / 末节点 text** 拼接，逐段
        advantage_char[last:last+len(text)] = node["advantage"] 广播赋值
        ——node 拼接恰好填满 response_text 时哨兵不残留，response_text
        更长则尾部留 -777；token 账：ones(len(input_ids)) * **-999**
        哨兵起底，**token advantage = advantage_char[offset_mapping[i][0]]**
        （多字符 token 只取起始字符的 advantage）；assert tokens[0] ==
        "▁"（GPT-2 BPE 首占位 token）；清理账：clean_response_input_ids
        = input_ids.reshape(-1)[1:]、clean_advantage_token =
        advantage_token[1:]（都剥首位 "▁"）。

与 to-gae / to-ppo-value-clip 划界：这里没有任何可学习价值网与 TD/λ 折扣
——value 本身就是 MC 回报均值，账目是纯树遍历 + 树形差分。
"""
import copy
import json
import random

import numpy as np

AGGREGATORS = ("none", "unique", "top_k_percent", "max_k_paths")
FILTERS = ("none", "successful", "non_zero_score",
           "non_zero_last_step_score", "zero_last_step_score",
           "non_zero_last_step_advantage")


def _is_leaf(node):
    return "children" not in node


def _compute_score(tree, gold, pred_map):
    """官方 compute_score：叶 score 从 node["answer"] 提取 pred，内部均值。"""

    def dfs(node):
        if "answer" in node:
            pred = pred_map.get(node["answer"])
            node["score"] = 0 if pred is None else int(pred == gold)
            node["is_correct_answer"] = node["score"]
            return node["score"]
        child_scores = [dfs(child) for child in node.get("children", [])]
        node["score"] = sum(child_scores) / len(child_scores) \
            if child_scores else 0
        return node["score"]

    dfs(tree)
    return tree


def _compute_advantage(root):
    """官方 compute_advantage：child.advantage = child.score − node.score。"""

    def dfs(node):
        if "answer" in node:
            return
        for child in node.get("children", []):
            child["advantage"] = child["score"] - node["score"]
            dfs(child)

    dfs(root)
    return root


def _extract_paths(tree, include_importance_weights=False,
                   repeat_early_stopped_paths=False,
                   branch_factor=None, max_depth=None):
    """官方 extract_paths_from_tree：deepcopy -> 权重写入 -> 浅叶重复 ->
    逐节点 pop("children") -> json round-trip。"""
    paths = []

    def dfs(node, depth, parent_node_chain):
        node_chain = parent_node_chain + [node]
        if branch_factor is not None and max_depth is not None:
            importance_weight = branch_factor ** (max_depth - depth)
        else:
            importance_weight = 1

        if include_importance_weights:
            node["importance_weight"] = (
                importance_weight if max_depth > depth else 1)

        if _is_leaf(node):
            row = {"node_chain": node_chain}
            if repeat_early_stopped_paths and max_depth > depth:
                paths.extend([row] * importance_weight)
            else:
                paths.append(row)

        for child in node.get("children", []):
            dfs(child, depth + 1, node_chain)
        # 与官方一致：序列化提速的副作用，顺手摘掉 children
        node.pop("children", None)

    tree = copy.deepcopy(tree)
    dfs(tree, 0, [])
    paths = json.loads(json.dumps(paths))
    return paths


def _aggregate(variant, paths, top_k_percent=None, keep_at_least_one=True,
               reduction="mean", k=None):
    """官方 PathAggregator 注册类分发。"""
    if variant == "none":
        return paths
    if variant == "unique":
        unique_paths = {json.dumps(p, sort_keys=True) for p in paths}
        return [json.loads(p) for p in unique_paths]
    if variant == "top_k_percent":
        assert 0 < top_k_percent <= 1
        if len(paths) == 0:
            return paths
        path_scores = []
        for path in paths:
            all_advantages = [node.get("advantage", 0)
                              for node in path["node_chain"]]
            if reduction == "mean":
                score = sum(all_advantages) / len(all_advantages)
            elif reduction == "sum":
                score = sum(all_advantages)
            else:
                raise ValueError("Unknown reduction: %s" % reduction)
            path_scores.append((path, score))
        path_scores = sorted(path_scores, key=lambda x: x[1], reverse=True)
        num_to_keep = int(len(path_scores) * top_k_percent)
        if keep_at_least_one:
            num_to_keep = max(num_to_keep, 1)
        return [path for path, _ in path_scores[:num_to_keep]]
    if variant == "max_k_paths":
        if len(paths) <= k:
            return paths
        return random.Random(42).sample(paths, k)
    raise ValueError("Unknown aggregator: %s" % variant)


def _filter(variant, path):
    """官方 PathFilter 注册类分发（单条 path -> bool）。"""
    last_node = path["node_chain"][-1]
    if variant == "none":
        return True
    if variant == "successful":
        if "answer" not in last_node:
            return False
        return last_node["is_correct_answer"]
    if variant == "non_zero_score":
        return path["node_chain"][1]["score"] > 0
    if variant == "non_zero_last_step_score":
        return last_node["score"] > 0
    if variant == "zero_last_step_score":
        return last_node["score"] == 0
    if variant == "non_zero_last_step_advantage":
        return last_node["advantage"] > 0
    raise ValueError("Unknown filter: %s" % variant)


def _advantage_split(node_chain, response_text, input_ids, offset_mapping,
                     tokens):
    """官方 convert_path_to_episode 的字符级 advantage 分摊 + token 清理。"""
    # ----- computing advantages -----
    advantage_char = np.ones(len(response_text)) * -777
    last_index = 0
    response_node_chain = node_chain[1:]
    for i, node in enumerate(response_node_chain):
        if i == 0:
            if len(response_node_chain) == 1:
                text = "\n" + node["text"]
            else:
                text = "\n" + node["text"] + "\n"
        elif i != len(response_node_chain) - 1:
            text = node["text"] + "\n"
        else:
            text = node["text"]
        advantage_char[last_index: last_index + len(text)] = node["advantage"]
        last_index += len(text)

    advantage_token = np.ones(len(input_ids)) * -999
    for i in range(len(advantage_token)):
        start_char_of_token = offset_mapping[i][0]
        advantage_token[i] = advantage_char[start_char_of_token]

    # ----- cleaning up the response -----
    assert tokens[0] == "▁"  # just repeating the official assertion
    ids_arr = np.asarray(input_ids)
    clean_response_input_ids = ids_arr.reshape(-1)[1:]
    clean_advantage_token = advantage_token[1:]
    return (advantage_char.tolist(), advantage_token.tolist(),
            clean_response_input_ids.tolist(), clean_advantage_token.tolist())


def solve(mode="score", tree=None, gold=None, pred_map=None,
          include_importance_weights=False, repeat_early_stopped_paths=False,
          branch_factor=None, max_depth=None,
          variant="none", paths=None, top_k_percent=None,
          keep_at_least_one=True, reduction="mean", k=None,
          node_chain=None, response_text=None, input_ids=None,
          offset_mapping=None, tokens=None):
    """统一入口（判题协议）：按 mode 分发到六个官方账段。"""
    if mode == "score":
        return _compute_score(copy.deepcopy(tree), gold, pred_map)
    if mode == "advantage":
        return _compute_advantage(copy.deepcopy(tree))
    if mode == "paths":
        return _extract_paths(tree, include_importance_weights,
                              repeat_early_stopped_paths, branch_factor,
                              max_depth)
    if mode == "aggregate":
        return _aggregate(variant, paths, top_k_percent, keep_at_least_one,
                          reduction, k)
    if mode == "filter":
        return _filter(variant, paths)
    if mode == "advantage_split":
        return _advantage_split(node_chain, response_text, input_ids,
                                offset_mapping, tokens)
    raise ValueError("Unknown mode: %s" % mode)
