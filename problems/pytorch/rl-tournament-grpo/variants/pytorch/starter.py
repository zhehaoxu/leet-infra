"""Tournament-GRPO (datawhalechina/agentic-rl-lab) -- binary format reward
+ knockout tournament reward ledger.

The judge injects a single ``solve``; dispatch by the ``mode`` keyword:

- mode="parse": stack-based transcript parsing (kwargs: text)
  -> [blocks, sorted(errors)]
- mode="answer": strict trailing <answer> extraction (kwargs: text)
- mode="format": 0/1 format reward + failure reasons + answer
  (kwargs: text)
- mode="normalize": group min-max, all-equal -> 0.5 (kwargs: scores)
- mode="tournament": knockout ledger (kwargs: query, answers, seed, pick)
  -> {raw_scores, rewards, matches}
- mode="score": full group scoring (kwargs: query, solutions, seed, pick)
  -> {results, log}

The LLM judge is parameterized as (seed, pick): pick="longest" always picks
the longest answer in a group, pick="first" picks index 0; seed drives the
official random.Random(seed) shuffles. Constants: M=4 repeats, G=2 players
per match (a 3-player group appears on odd candidate counts), +1 per win.
"""
