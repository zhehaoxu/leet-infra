"""ARPO（Agentic Reinforced Policy Optimization，arXiv:2507.19849，
RUC-NLPIR/ARPO）熵感知 rollout 分支账。

ARPO 的核心主张：多轮工具调用 agent RL 里，环境反馈后的 token 分布
不确定性会骤升——**对高熵时刻自适应分支采样**比均匀多采样更省预算。
官方 vllm_rollout_with_tools.py 的确定性账（vLLM 引擎/工具执行器/日志
剥除，rollout 输出与随机源全入参），solve(mode=...) 五个入口：

    mode="entropy"（Entropy Variation Monitoring 段）：
        outputs（每 rollout 含 per_token_topk：每 token 的 top-10
        logprob 列表）+ active_indices + vocab_size + initial_entropy_dict
        -> (current_entropy_dict, initial_entropy_dict)。
        逐 rollout 取**前 min(20, len(tokens)) 个 token** 的 top-10
        logprobs 扁平拼接（某 token 缺 logprobs 用最后一个的），熵 =
        −Σ p·l（p=exp(l)、对 top-k 截断分布求和，logprobs 空 → 0），
        归一化除 **log(vocab_size)**；out_idx 首次出现时把当前熵写入
        initial_entropy_dict（**此后不再更新**——delta 是对首次观测的
        漂移）。

    mode="branch"（Entropy-based Adaptive Beaming）：
        active_by_sample（orig_sample -> 活跃 rollout 下标列表）+
        current/initial_entropy_dict + rollouts_per_sample +
        num_samples + beam_size + branch_probability + entropy_weight +
        rand_fn（无参随机源）-> (new_branches, rollouts_per_sample)。
        每样本 remaining_slots = num_samples − rollouts_per_sample，
        ≤0 跳过；每活跃 idx branches_per_idx = **min(beam_size − 1,
        remaining_slots − branches_created)**，≤0 break；每分支
        prob = clamp(rand() − entropy_weight × (entropy_now −
        entropy_init), 0, 1)，**prob > branch_probability → 跳过**
        （熵升 → prob 降 → 更易分支：环境反馈后不确定性升高才分叉）；
        分支记 (source_idx, orig_sample)、rollouts_per_sample += 1、
        branches_created 达 remaining_slots 即 break。

    mode="refill"（不活跃样本补支）：
        active_by_sample + rollouts_per_sample + num_samples +
        sample_to_indices -> (refills, rollouts_per_sample)。
        orig 不在 active_by_sample 且未满 → 每轮只补 **1** 支，模板取
        sample_to_indices[orig][0]（官方用其 init_inputs 重开：curr=
        init、masks 空、counter 0——该语义由调用方落地，此处记
        (orig_sample, template_idx)）。

    mode="round"（回合控制）：
        finish_reason + stop_reason + stop_sequences + call_counter +
        tool_call_limit + eos_token_id + response_len + max_len ->
        (action, new_counter, extra_tokens)。工具调用
        （finish=="stop" 且 stop_reason ∈ stop_sequences）计数器未满 →
        ("tool", counter+1, [])；**满额 → ("limit_eos", counter,
        [eos])**（追加 EOS 终止、call_limit_reached 统计由调用方累加）；
        finish=="length" 且 response_len < max_len → ("continue", …)
        否则 stop；finish=="stop"（EOS）→ stop。

    mode="reorg"（输出重组）：
        batch_size + sample_to_indices + num_samples + curr_inputs +
        result_masks + prompt_lens -> (output_sequences,
        output_result_masks)。每样本取**前 num_samples 个**下标（不足
        **复制最后一个**补齐）；response = curr_inputs[idx][prompt_len:]；
        result_masks 与 token 一一对应（环境回填段是 0、模型生成段是 1，
        由回合账维护，此处只透传切片）。

与 to-grpo-advantage（组级归一化）划界：ARPO 的预算分配发生在 rollout
阶段（同一 prompt 的样本数动态向高熵分支倾斜），advantage 侧仍是组内
标准化；本题只收 rollout 侧的熵监测与分支账。
"""
import math

# top-k logprobs per token for entropy estimation（官方 logprobs=10）
ENTROPY_TOPK = 10
# 参与熵估计的每 rollout 最大 token 数（官方固定 20）
ENTROPY_TOKEN_WINDOW = 20


def calc_entropy(logprobs):
    """官方 _calc_entropy：top-k 截断分布的 −Σ p·l；空 -> 0。"""
    if not logprobs:
        return 0.0
    p_list = [math.exp(l) for l in logprobs]
    entropy = -sum(p * l for p, l in zip(p_list, logprobs))
    return entropy


def _entropy_monitor(outputs, active_indices, initial_entropy_dict,
                     vocab_size):
    """官方 Entropy Variation Monitoring：top-10 扁平 -> 归一化 -> 初始熵。"""
    current_entropy_dict = {}
    for i, out_idx in enumerate(active_indices):
        output = outputs[i]
        logprobs = []
        tokens = output["token_ids"]
        per_token = output["per_token_topk"]
        for j in range(min(ENTROPY_TOKEN_WINDOW, len(tokens))):
            try:
                logprob_info = per_token[j]
            except Exception:
                logprob_info = per_token[-1]
            token_list = list(logprob_info.values())
            token_logprobs = [token for token in token_list]
            logprobs.extend(token_logprobs)
        if logprobs:
            entropy = calc_entropy(logprobs) / math.log(vocab_size)
        else:
            entropy = 0.0
        current_entropy_dict[out_idx] = entropy
        if out_idx not in initial_entropy_dict:
            initial_entropy_dict[out_idx] = entropy
    return current_entropy_dict


def _adaptive_branch(active_by_sample, current_entropy_dict,
                     initial_entropy_dict, rollouts_per_sample, num_samples,
                     beam_size, branch_probability, entropy_weight, rand_fn):
    """官方 Entropy-based Adaptive Beaming：熵 delta 门控的分支分配。"""
    new_branches = []
    for orig_sample, active_idxs in active_by_sample.items():
        remaining_slots = num_samples - rollouts_per_sample[orig_sample]
        if remaining_slots <= 0:
            continue
        branches_created = 0
        for source_idx in active_idxs:
            branches_per_idx = min(beam_size - 1,
                                   remaining_slots - branches_created)
            if branches_per_idx <= 0:
                break
            for _ in range(branches_per_idx):
                entropy_now = current_entropy_dict.get(source_idx, 0.0)
                entropy_init = initial_entropy_dict.get(source_idx, 0.0)
                entropy_delta = entropy_now - entropy_init
                prob = rand_fn() - entropy_weight * entropy_delta
                prob = max(0.0, min(1.0, prob))
                if prob > branch_probability:
                    continue
                new_branches.append((source_idx, orig_sample))
                rollouts_per_sample[orig_sample] += 1
                branches_created += 1
            if branches_created >= remaining_slots:
                break
    return new_branches, rollouts_per_sample


def _refill_inactive(active_by_sample, rollouts_per_sample, num_samples,
                     sample_to_indices):
    """官方不活跃样本补支：每轮 1 支、模板取该样本首个下标。"""
    refills = []
    for orig_sample in range(max(sample_to_indices) + 1
                             if sample_to_indices else 0):
        if orig_sample not in active_by_sample and \
                rollouts_per_sample[orig_sample] < num_samples:
            branches_to_add = min(1,
                                  num_samples - rollouts_per_sample[orig_sample])
            if branches_to_add <= 0:
                continue
            source_idx = sample_to_indices[orig_sample][0]
            for _ in range(branches_to_add):
                refills.append((orig_sample, source_idx))
                rollouts_per_sample[orig_sample] += 1
    return refills, rollouts_per_sample


def _round_control(finish_reason, stop_reason, stop_sequences, call_counter,
                   tool_call_limit, eos_token_id, response_len, max_len):
    """官方回合控制：工具限额 / EOS / length 三分支。"""
    is_tool_call = finish_reason == "stop" and stop_reason in stop_sequences
    if is_tool_call:
        if call_counter < tool_call_limit:
            return ("tool", call_counter + 1, [])
        return ("limit_eos", call_counter, [eos_token_id])
    if finish_reason == "length":
        if response_len < max_len:
            return ("continue", call_counter, [])
        return ("stop", call_counter, [])
    if finish_reason == "stop":
        return ("stop", call_counter, [])
    return ("stop", call_counter, [])


def _reorganize_outputs(batch_size, sample_to_indices, num_samples,
                        curr_inputs, result_masks, prompt_lens):
    """官方输出重组：每样本前 num_samples 个、不足复制最后一个。"""
    output_sequences = []
    output_result_masks = []
    for i in range(batch_size):
        sample_indices = sample_to_indices.get(i, [])
        selected_indices = sample_indices[:num_samples]
        while len(selected_indices) < num_samples:
            if selected_indices:
                selected_indices.append(selected_indices[-1])
            else:
                break
        for idx in selected_indices:
            output_sequences.append(curr_inputs[idx][prompt_lens[i]:])
            output_result_masks.append(result_masks[idx])
    return output_sequences, output_result_masks


def solve(mode="entropy", outputs=None, active_indices=None,
          initial_entropy_dict=None, vocab_size=None,
          active_by_sample=None, current_entropy_dict=None,
          rollouts_per_sample=None, num_samples=None, beam_size=1,
          branch_probability=0.5, entropy_weight=0.5, rand_fn=None,
          sample_to_indices=None, finish_reason=None, stop_reason=None,
          stop_sequences=None, call_counter=0, tool_call_limit=5,
          eos_token_id=None, response_len=0, max_len=0, batch_size=0,
          curr_inputs=None, result_masks=None, prompt_lens=None):
    """统一入口（判题协议）：按 mode 分发到五个官方账段。"""
    if mode == "entropy":
        cur = _entropy_monitor(outputs, active_indices, initial_entropy_dict,
                               vocab_size)
        return cur, initial_entropy_dict
    if mode == "branch":
        return _adaptive_branch(active_by_sample, current_entropy_dict,
                                initial_entropy_dict, rollouts_per_sample,
                                num_samples, beam_size, branch_probability,
                                entropy_weight, rand_fn)
    if mode == "refill":
        return _refill_inactive(active_by_sample, rollouts_per_sample,
                                num_samples, sample_to_indices)
    if mode == "round":
        return _round_control(finish_reason, stop_reason, stop_sequences,
                              call_counter, tool_call_limit, eos_token_id,
                              response_len, max_len)
    if mode == "reorg":
        return _reorganize_outputs(batch_size, sample_to_indices,
                                   num_samples, curr_inputs, result_masks,
                                   prompt_lens)
    raise ValueError("Unknown mode: %s" % mode)
