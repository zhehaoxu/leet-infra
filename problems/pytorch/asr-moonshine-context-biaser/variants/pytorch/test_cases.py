"""Moonshine context biaser test cases — judge-style: check(solve) + string CHECKS.
Values pinned by the official context-biaser-test.cpp doctest."""
import math

LOG2 = math.log(2.0)
LOG3 = math.log(3.0)


def check_variants(solve):
    # mid-sentence and utterance-initial forms both need registering
    assert solve(mode="variants", term="Kubernetes") == ["Kubernetes", " Kubernetes"]
    # surrounding whitespace is not meaningful
    assert solve(mode="variants", term="  Kubernetes  ")[0] == "Kubernetes"
    # blank term registers nothing
    assert solve(mode="variants", term="   ") == []
    # a caller that anchors the term itself gets exactly what they asked for
    assert solve(mode="variants", term="\u2581Kubernetes") == ["\u2581Kubernetes"]


def check_build(solve):
    t = solve(mode="build", sequences=[[10, 11, 12]])
    assert t["sequence_count"] == 1 and t["max_depth"] == 3
    assert len(t["children"]) == 4 and t["depths"] == [0, 1, 2, 3]
    assert t["children"][0] == {10: 1} and t["children"][1] == {11: 2}
    # shared prefixes reuse nodes: [10,11] and [10,12] -> 4 nodes, depth 2
    t2 = solve(mode="build", sequences=[[10, 11], [10, 12]])
    assert len(t2["children"]) == 4 and t2["max_depth"] == 2
    assert t2["children"][1] == {11: 2, 12: 3}
    # a duplicate sequence raises the count but adds no nodes
    t3 = solve(mode="build", sequences=[[10, 11], [10, 11]])
    assert t3["sequence_count"] == 2 and len(t3["children"]) == 3
    # empty sequences are not registered
    t4 = solve(mode="build", sequences=[])
    assert t4["sequence_count"] == 0 and t4["max_depth"] == 0
    assert t4["children"] == [{}] and t4["depths"] == [0]


def check_bonus(solve):
    boost = 2.0
    # depth 0 contributes nothing; the ramp is boost * (1 + ln depth)
    assert solve(mode="bonus_for_depth", depth=0, boost=boost) == 0.0
    assert abs(solve(mode="bonus_for_depth", depth=1, boost=boost) - 2.0) < 1e-12
    assert abs(solve(mode="bonus_for_depth", depth=2, boost=boost) - 2.0 * (1 + LOG2)) < 1e-12
    assert abs(solve(mode="bonus_for_depth", depth=3, boost=boost) - 2.0 * (1 + LOG3)) < 1e-12
    # completion is rewarded more than the entry
    d1 = solve(mode="bonus_for_depth", depth=1, boost=boost)
    d3 = solve(mode="bonus_for_depth", depth=3, boost=boost)
    assert d3 > d1


def check_apply(solve):
    vocab = 64
    boost = 2.0
    # only the entry token is a candidate before anything is emitted
    t = solve(mode="build", sequences=[[10, 11, 12]])
    out = solve(mode="apply", trie=t, active=[0], boost=boost, vocab_size=vocab)
    assert out == {10: 2.0}, out
    # shared prefixes are not stacked: both terms start with 10, boosted once
    t2 = solve(mode="build", sequences=[[10, 11], [10, 12]])
    out2 = solve(mode="apply", trie=t2, active=[0], boost=3.0, vocab_size=vocab)
    assert out2 == {10: 3.0}, out2
    # a token proposed by two paths takes the larger bonus, not the sum
    t3 = solve(mode="build", sequences=[[10, 11], [11, 20]])
    act = solve(mode="advance", trie=t3, active=[0], token=10)
    out3 = solve(mode="apply", trie=t3, active=act, boost=3.0, vocab_size=vocab)
    assert abs(out3[11] - 3.0 * (1 + LOG2)) < 1e-12, out3
    assert abs(out3[10] - 3.0) < 1e-12
    # zero boost is an exact no-op
    out4 = solve(mode="apply", trie=t, active=[0], boost=0.0, vocab_size=vocab)
    assert out4 == {10: 0.0}
    # out-of-range tokens are skipped entirely
    t5 = solve(mode="build", sequences=[[70, 11]])
    out5 = solve(mode="apply", trie=t5, active=[0], boost=2.0, vocab_size=vocab)
    assert out5 == {}
    # an empty biaser changes nothing
    t6 = solve(mode="build", sequences=[])
    assert solve(mode="apply", trie=t6, active=[0], boost=2.0, vocab_size=vocab) == {}


def check_advance(solve):
    t = solve(mode="build", sequences=[[10, 11, 12]])
    # walking down the path activates the deeper node
    act = solve(mode="advance", trie=t, active=[0], token=10)
    assert act == [0, 1]
    # abandoning a path drops back to the root, but the term can restart
    act2 = solve(mode="advance", trie=t, active=act, token=42)
    assert act2 == [0]
    # a term can start part-way through another: [10,11] and [11,20]
    t2 = solve(mode="build", sequences=[[10, 11], [11, 20]])
    a1 = solve(mode="advance", trie=t2, active=[0], token=10)
    a2 = solve(mode="advance", trie=t2, active=a1, token=11)
    # after "10 11" the node inside [10,11] matched fully AND the root matched
    # 11 as the start of [11,20]
    assert a2 == [0, 2, 3] or sorted(a2) == [0, 2, 3], a2
    assert solve(mode="bonus_for_token", trie=t2, active=a2, boost=2.0,
                 token=20) > 0.0
    # the walk after 10 keeps both the root and the node inside [10,11]
    assert sorted(a1) == [0, 1]
    # an empty biaser never moves
    t3 = solve(mode="build", sequences=[])
    assert solve(mode="advance", trie=t3, active=[0], token=10) == [0]


def check_walk(solve):
    vocab = 64
    boost = 2.0
    t = solve(mode="build", sequences=[[10, 11, 12]])
    # bonuses land BEFORE the token is chosen; the walk advances over the token
    steps = solve(mode="bias_walk", trie=t, boost=boost,
                  tokens=[10, 11, 12], vocab_size=vocab)
    assert steps[0] == {10: 2.0}
    assert abs(steps[1][11] - 2.0 * (1 + LOG2)) < 1e-12, steps[1]
    assert abs(steps[2][12] - 2.0 * (1 + LOG3)) < 1e-12, steps[2]
    # an unrelated token between path tokens drops the partial match
    steps2 = solve(mode="bias_walk", trie=t, boost=boost,
                   tokens=[10, 42, 10], vocab_size=vocab)
    assert steps2[0] == {10: 2.0}
    # at the 42 step the deeper node is still active (it dies on the 42 itself)
    assert steps2[1] == {10: 2.0, 11: 2.0 * (1 + LOG2)}
    assert steps2[2] == {10: 2.0}  # back at the root: only entries proposed
    # the walk after 10 (no mismatch) would have offered the depth-2 bonus
    steps3 = solve(mode="bias_walk", trie=t, boost=boost,
                   tokens=[10, 42], vocab_size=vocab)
    assert abs(steps3[1][11] - 2.0 * (1 + LOG2)) < 1e-12
    # empty biaser: every step proposes nothing
    t4 = solve(mode="build", sequences=[])
    assert solve(mode="bias_walk", trie=t4, boost=boost,
                 tokens=[1, 2, 3], vocab_size=vocab) == [{}, {}, {}]


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
