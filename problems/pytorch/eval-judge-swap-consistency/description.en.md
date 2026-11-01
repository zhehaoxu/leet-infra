# LLM-as-judge position-bias ledger: swapped games, disagreement-as-tie and win-rate aggregation

Pairwise judging by an LLM is biased toward the answer in position A (position bias). The MT-Bench official implementation (FastChat `fastchat/llm_judge`) plays **two games per question**: g1 with (answer_1, answer_2) placement and g2 with swapped placement; a real win/loss requires both games to agree, and disagreement counts as a tie. This problem implements the full ledger: judgment parsing → swap mapping → per-match resolution → aggregation.

## Judgment parsing ("[[A]]" output format)

Check each game's judgment text **in order** (official if/elif chain, independent of where the marks appear in the text):

1. contains `[[A]]` → side A wins
2. else contains `[[B]]` → side B wins
3. else contains `[[C]]` → tie
4. else → error

Matching is case-sensitive (`[[a]]` is not a mark). When a text contains multiple marks, **the earliest-checked mark wins** — the official chain-order quirk.

## Placement mapping (swap ledger)

- g1: A → model_1, B → model_2
- g2 (swapped placement): A → model_2, B → model_1
- tie / error are not mapped and stay as-is

## Per-match resolution (official show_result condition)

1. If either game is an **error → the whole match is dropped** (outcome = "error", counted nowhere). Priority note: error-dropping precedes the tie rule — a tie in g1 with an unparseable g2 still drops the match.
2. g1 is a tie, **or** the two mapped games disagree (the typical position-bias shape: after swapping, the judge still picks position A, so the two games each pick a different model) → tie (both models get one tie).
3. Both games agree and neither is a tie → win/loss (winner gets a win, loser a loss).

## Statistics and win rate

```
win_rate_adjusted = (win + 0.5 * tie) / (win + loss + tie)
```

A tie counts as half a win and half a loss; models with a zero denominator get win_rate_adjusted = 0.0. stats covers every model listed in `models` (models that never appear get all-zero rows).

## Input / Output

```python
solve(matches, models)
# matches: [{"question_id", "model_1", "model_2", "g1_judgment", "g2_judgment"}, ...]
# returns:
{
  "matches": [  # same length and order as input
    {"question_id", "g1_winner", "g2_winner", "outcome"}
    # g1_winner/g2_winner are mapped: "model_1"/"model_2"/"tie"/"error"
  ],
  "stats": {model: {"win", "loss", "tie", "win_rate_adjusted"}}
}
```

## Example

```
Input:  matches = [{"question_id": 1, "model_1": "alpha", "model_2": "beta",
                    "g1_judgment": "x [[A]] y",        # position A = alpha wins
                    "g2_judgment": "y [[A]] x"}],      # position A = beta wins
        models = ["alpha", "beta"]
Output: matches[0] = {"g1_winner": "alpha", "g2_winner": "beta", "outcome": "tie"}
        (position bias: the judge still picks position A after swapping → tie)
        stats = {"alpha": {"win":0,"loss":0,"tie":1,"win_rate_adjusted":0.5}, ...}
```
