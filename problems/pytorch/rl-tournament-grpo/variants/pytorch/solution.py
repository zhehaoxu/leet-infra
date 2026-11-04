"""Tournament-GRPO (datawhalechina/agentic-rl-lab experiments/tournament-grpo,
adapted from PuffYang/Tournament_GRPO) -- binary format reward + knockout
tournament reward, transcribed from reward.py.

The judge injects a single ``solve``; all entry points are dispatched by the
``mode`` keyword. The LLM judge is parameterized as a deterministic pick
strategy ("longest" = the judge always picks the longest answer in the
group, "first" = always index 0) plus a seed driving the official
random.Random(seed) shuffles, so the whole ledger is deterministic.

Official constants kept verbatim:
- MAX_ROLLOUT_CHARS = 120000, TOURNAMENT_REPEATS = 4 (M, fresh draws each
  repeat), PLAYERS_PER_MATCH = 2 (G, one winner per match, +1 point)
- tag errors are reported in the author's fixed order
  (unknown_tags, invalid_attrs, broken_xml, stray_text_outside_tags)
- transcripts parse with a stack; a close tag that mismatches the stack top
  falls back to the nearest same-name opener and silently drops the
  in-between unclosed tags
- only call_tool may carry attributes and must carry name="..."
- <answer> must be unique, non-empty, tag-free and trail the transcript
- structure: think first, then segments of >=1 google_search cycle plus any
  number of browse_webpage cycles, then the answer; one cycle =
  call_tool -> tool_output -> think, all three non-empty
- group reward = min-max normalized tournament reward (all-equal -> 0.5
  each) + binary format reward; candidates without an extracted answer
  still enter the tournament with "" but get their tournament reward zeroed
"""
import re


MAX_ROLLOUT_CHARS = 120000
TOURNAMENT_REPEATS = 4
PLAYERS_PER_MATCH = 2

TAG_PATTERN = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_]*)([^>]*)>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ATTRIBUTE_PATTERN = re.compile(r'\s+([A-Za-z_][A-Za-z0-9_]*)="([^"]*)"')
ALLOWED_TAGS = {"think", "call_tool", "tool_output", "snippet", "webpage",
                "answer"}
TOOL_NAMES = {"google_search", "browse_webpage"}
TAG_ERRORS = ("tag:unknown_tags", "tag:invalid_attrs", "tag:broken_xml",
              "structure:stray_text_outside_tags")


def parse_attrs(tag, text):
    """Official parse_attrs: only call_tool may carry attrs and needs name."""
    attrs, cursor = {}, 0
    for match in ATTRIBUTE_PATTERN.finditer(text):
        key, value = match.groups()
        if text[cursor:match.start()].strip() or key in attrs:
            return attrs, False
        attrs[key] = value
        cursor = match.end()
    allowed = {"name"} if tag == "call_tool" else set()
    valid = not text[cursor:].strip() and not (attrs.keys() - allowed)
    return attrs, valid and (tag != "call_tool" or "name" in attrs)


def parse_transcript(text):
    """Official parse_transcript: top-level blocks + tag errors."""
    blocks, stack, errors = [], [], set()
    cursor = 0
    for match in TAG_PATTERN.finditer(text):
        if not stack and text[cursor:match.start()].strip():
            errors.add("structure:stray_text_outside_tags")
        closing, tag, attributes = match.groups()
        cursor = match.end()
        if tag not in ALLOWED_TAGS:
            errors.add("tag:unknown_tags")
        if not closing:
            attrs, valid = parse_attrs(tag, attributes)
            if not valid:
                errors.add("tag:invalid_attrs")
            stack.append((tag, attrs, match.end()))
            continue
        if attributes.strip():
            errors.add("tag:invalid_attrs")
        if stack and stack[-1][0] == tag:
            opened = stack.pop()
        else:
            errors.add("tag:broken_xml")
            index = max((i for i, (open_tag, _, _) in enumerate(stack)
                         if open_tag == tag), default=None)
            if index is None:
                continue
            opened, stack = stack[index], stack[:index]
        if not stack:
            open_tag, attrs, start = opened
            blocks.append({"tag": open_tag, "attrs": attrs,
                           "content": text[start:match.start()]})
    if not stack and text[cursor:].strip():
        errors.add("structure:stray_text_outside_tags")
    if stack:
        errors.add("tag:broken_xml")
    return blocks, errors


def extract_answer(text):
    """Official extract_answer: unique, non-empty, tag-free, trailing."""
    matches = list(ANSWER_PATTERN.finditer(text))
    if len(matches) != 1:
        return ""
    answer = matches[0].group(1).strip()
    if not answer or TAG_PATTERN.search(answer) or \
            text[matches[0].end():].strip():
        return ""
    return answer


def _tool_name(block):
    return block["attrs"].get("name", "").strip()


def _is_tool_cycle(blocks, index, name):
    if index + 2 >= len(blocks):
        return False
    call, observation, think = blocks[index:index + 3]
    return (call["tag"] == "call_tool" and _tool_name(call) == name and
            bool(call["content"].strip()) and
            observation["tag"] == "tool_output" and
            bool(observation["content"].strip()) and
            think["tag"] == "think" and bool(think["content"].strip()))


def check_structure(blocks):
    """Official check_structure: think -> research segments -> answer."""
    if blocks[0]["tag"] != "think" or not blocks[0]["content"].strip():
        return "structure:must_start_with_non_empty_think"
    if len(blocks) <= 2:
        return "structure:must_have_at_least_one_tool_cycle"
    index = 1
    while index < len(blocks) - 1:
        if not _is_tool_cycle(blocks, index, "google_search"):
            return "structure:each_segment_must_start_with_google_search_cycle"
        while _is_tool_cycle(blocks, index, "google_search"):
            index += 3
        while _is_tool_cycle(blocks, index, "browse_webpage"):
            index += 3
    return None


def analyze_format(text):
    """Official analyze_format: 0/1 reward + failure reasons + answer."""
    blocks, errors = parse_transcript(text)
    failures = [reason for reason in TAG_ERRORS if reason in errors]
    if any(block["tag"] == "call_tool" and
           _tool_name(block) not in TOOL_NAMES for block in blocks):
        failures.append("tool:unknown_tool_name")
    answer = extract_answer(text)
    if not answer:
        failures.append("answer:invalid_or_missing_final_answer")
    if not failures:
        structure_failure = check_structure(blocks)
        if structure_failure:
            failures.append(structure_failure)
    return {"format_reward": float(not failures),
            "failure_reasons": failures, "rollout_answer": answer}


def normalize_scores(scores):
    """Official normalize_scores: group min-max; all-equal -> 0.5 each."""
    low, high = min(scores), max(scores)
    return [0.5] * len(scores) if high == low else \
        [(value - low) / (high - low) for value in scores]


def run_tournament(query, rubrics, answers, seed, pick):
    """Official run_tournament with the judge parameterized as (seed, pick).

    M=4 repeats with fresh shuffles, G=2 players per match (a 3-player group
    appears when the candidate count is odd), round-robin grouping
    candidates[i::num_groups], winner +1.
    """
    import random
    scores = [0.0] * len(answers)
    matches = []
    rng = random.Random(seed)
    for repeat in range(1, TOURNAMENT_REPEATS + 1):
        candidates = list(range(len(answers)))
        round_index = 0
        while len(candidates) > 1:
            round_index += 1
            rng.shuffle(candidates)
            num_groups = len(candidates) // PLAYERS_PER_MATCH
            groups = [candidates[i::num_groups] for i in range(num_groups)]
            winners = []
            for group in groups:
                group_answers = [answers[i] for i in group]
                if pick == "longest":
                    local_index = max(range(len(group_answers)),
                                      key=lambda i: len(group_answers[i]))
                else:
                    local_index = 0
                output = "stub-output"
                winner = group[local_index]
                scores[winner] += 1.0
                winners.append(winner)
                matches.append({"repeat": repeat, "round": round_index,
                                "candidates": group, "winner": winner,
                                "output": output})
            candidates = winners
    return {"raw_scores": scores, "rewards": normalize_scores(scores),
            "matches": matches}


def score_group(query, solutions, seed, pick):
    """Official score_group: total = tournament reward + format reward.

    No-answer candidates enter with "" but get their tournament reward
    zeroed after normalization.
    """
    formats = [analyze_format(text) for text in solutions]
    answers = [info["rollout_answer"][:MAX_ROLLOUT_CHARS]
               for info in formats]
    tournament = run_tournament(query.strip(), [], answers, seed, pick)
    results = []
    for info, answer, reward, raw in zip(formats, answers,
                                         tournament["rewards"],
                                         tournament["raw_scores"]):
        reward = reward if answer else 0.0
        results.append({**info, "rollout_answer": answer,
                        "score": reward + info["format_reward"],
                        "tournament_reward": reward,
                        "tournament_raw_score": raw})
    log = {"answers": answers, "raw_scores": tournament["raw_scores"],
           "tournament_rewards": [r["tournament_reward"] for r in results],
           "format_rewards": [r["format_reward"] for r in results],
           "matches": tournament["matches"]}
    return results, log


def solve(mode=None, **kw):
    if mode == "parse":
        blocks, errors = parse_transcript(kw["text"])
        return [blocks, sorted(errors)]
    if mode == "answer":
        return extract_answer(kw["text"])
    if mode == "format":
        return analyze_format(kw["text"])
    if mode == "normalize":
        return normalize_scores(kw["scores"])
    if mode == "tournament":
        res = run_tournament(kw["query"], kw.get("rubrics", []),
                             kw["answers"], kw["seed"], kw["pick"])
        return {"raw_scores": res["raw_scores"], "rewards": res["rewards"],
                "matches": res["matches"]}
    if mode == "score":
        results, log = score_group(kw["query"], kw["solutions"], kw["seed"],
                                   kw["pick"])
        return {"results": results, "log": log}
    raise ValueError("unknown mode: %s" % mode)
