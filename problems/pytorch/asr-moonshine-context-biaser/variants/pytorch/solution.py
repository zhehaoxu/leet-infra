"""Moonshine context biaser — trie walk over subword token IDs.

Reproduces the pure-function core of moonshine-ai/moonshine core/context-biaser.cpp
(decode-time contextual biasing toward key terms, no retraining):

- build: register tokenized key terms into a prefix trie (root = node 0). Shared
  prefixes reuse existing nodes; a duplicate sequence raises sequence_count but
  adds no nodes; empty sequences are not registered (an empty biaser is a
  no-op). depths[i] is the node depth (root = 0).
- variants: spellings to register for one key term — trimmed; empty -> [];
  already anchored with the U+2581 word-start marker -> [term]; otherwise
  [term, " " + term] so both mid-sentence and utterance-initial forms match.
- bonus_for_depth: boost * (1 + ln(depth)) for depth > 0, else 0. The ramp
  rewards completing a term more than starting one (greedy decoding cannot
  take a token back, so a flat bonus would fire on bare prefixes).
- apply: every child of an active node is proposed with the bonus of its own
  depth (bonus_for_depth(node.depth + 1)); out-of-range tokens are skipped;
  overlapping terms proposing the same token take the LARGER bonus, never
  stacked. Returns {token: bonus}.
- advance: next_active = [root] + children matched from the active nodes — the
  root always stays active so a term can start part-way through another, and a
  mismatched token drops the walk back to the root.
- bonus_for_token: the bonus apply() would currently add to one token (max over
  active nodes).
- bias_walk: decode-loop simulation — reset to [0], then per emitted token:
  record the apply() bonuses, then advance over the token. Officially the
  bonuses land just before the argmax and the walk advances over the chosen
  token (in speculative verification the walk follows the teacher-forced
  prefix, not the predictions).

All values here are float64 math on small ints; the official C++ tests pin the
same numbers (boost 2.0 -> depth-1 bonus 2.0, depth-2 2*(1+ln 2), ...).
"""

import math


def _bonus_for_depth(depth, boost):
    if depth <= 0:
        return 0.0
    return boost * (1.0 + math.log(depth))


def solve(mode, **kw):
    if mode == "variants":
        marker = "\u2581"
        trimmed = kw["term"].strip()
        if trimmed == "":
            return []
        if trimmed.startswith(marker):
            return [trimmed]
        return [trimmed, " " + trimmed]

    if mode == "build":
        sequences = kw["sequences"]
        children = [{}]
        depths = [0]
        max_depth = 0
        sequence_count = 0
        for tokens in sequences:
            if not tokens:
                continue
            node_index = 0
            for token in tokens:
                child = children[node_index].get(token)
                if child is not None:
                    node_index = child
                    continue
                child_depth = depths[node_index] + 1
                child_index = len(children)
                children[node_index][token] = child_index
                children.append({})
                depths.append(child_depth)
                max_depth = max(max_depth, child_depth)
                node_index = child_index
            sequence_count += 1
        return {"children": children, "depths": depths, "max_depth": max_depth,
                "sequence_count": sequence_count}

    if mode == "bonus_for_depth":
        return _bonus_for_depth(kw["depth"], kw["boost"])

    if mode == "apply":
        trie, active = kw["trie"], kw["active"]
        boost, vocab_size = kw["boost"], kw["vocab_size"]
        if trie["sequence_count"] == 0:
            return {}
        depths, children = trie["depths"], trie["children"]
        pending = {}
        for node_index in active:
            bonus = _bonus_for_depth(depths[node_index] + 1, boost)
            for token in children[node_index]:
                if token < 0 or token >= vocab_size:
                    continue
                if token in pending:
                    pending[token] = max(pending[token], bonus)
                else:
                    pending[token] = bonus
        return pending

    if mode == "advance":
        trie, active, token = kw["trie"], kw["active"], kw["token"]
        if trie["sequence_count"] == 0:
            return list(active)
        children = trie["children"]
        next_active = [0]
        for node_index in active:
            child = children[node_index].get(token)
            if child is not None:
                next_active.append(child)
        return next_active

    if mode == "bonus_for_token":
        trie, active = kw["trie"], kw["active"]
        boost, token = kw["boost"], kw["token"]
        depths, children = trie["depths"], trie["children"]
        best = 0.0
        for node_index in active:
            child = children[node_index].get(token)
            if child is not None:
                best = max(best, _bonus_for_depth(depths[child], boost))
        return best

    if mode == "bias_walk":
        trie, boost = kw["trie"], kw["boost"]
        tokens, vocab_size = kw["tokens"], kw["vocab_size"]
        active = [0]
        steps = []
        for token in tokens:
            steps.append(solve(mode="apply", trie=trie, active=active,
                               boost=boost, vocab_size=vocab_size))
            active = solve(mode="advance", trie=trie, active=active, token=token)
        return steps

    raise ValueError(f"unknown mode: {mode}")


CHECKS = [
    "check_variants",
    "check_build",
    "check_bonus",
    "check_apply",
    "check_advance",
    "check_walk",
]


def check(solve):
    for name in CHECKS:
        globals()[name](solve)
