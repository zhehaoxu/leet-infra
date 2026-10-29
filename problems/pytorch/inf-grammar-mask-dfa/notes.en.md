# Solution Notes · Regex-Constrained Token Mask

## Origin

- Outlines: *Efficient Guided Generation for Large Language Models* (Willard & Louf, arXiv:2307.09702) — compiles a regex / JSON Schema into a finite automaton (an index automaton) and masks the vocabulary state by state; it is the most cited implementation of FSA-constrained decoding.
- xgrammar (MLC) and llama.cpp's GBNF stack: the same idea for JSON-Schema / context-free grammars, using the current state as a cache key to precompute the token mask for every state.

This problem isolates the regex subset and the vocabulary mask. JSON-Schema normalisation, tokenizer alignment and the grammar stack of a CFG are out of scope; the automaton state itself is the whole exercise.

## Key points and pitfalls

**A state is "which positions the last consumed character landed on".** The Glushkov (position) automaton numbers every literal position of `pattern` and derives `first` / `last` / `follow`: the first character picks a matching position from `first`, each later step picks a successor from `follow`. The initial state is not an ordinary state — if you treat `first` as "positions already read" and then follow `follow`, the first character can never match (an `a*` pattern would be declared entirely illegal).

**Legal ≠ complete ≠ stoppable.** `mask` asks "can this still be completed", `complete` asks "does appending it finish the string", `accept` asks "is the prefix already a full match". The three separate on `a(b|c)*d`: after `"a"` and `"b"`, `mask` is true while `complete` is false; after `"d"` both are true; with `prefix="abc"` under `abc`, `accept` is true but every token's `mask` is false (a final state has no outgoing edges).

**Nullable patterns.** `a*` and `x?` contain ε paths. For an empty prefix `accept` must be true; if empty tokens were allowed, one would be legal exactly when the prefix itself is acceptable (this problem declares tokens non-empty).

**Multi-character tokens advance one character at a time.** A token is a string, not one alphabet symbol: any character that kills the state kills the whole token, and the landing state decides `complete`. Under `[ab]+c`, `"cc"` dies on the first character while `"abc"` lands exactly on a full match.

**Characters outside the grammar.** Vocabularies contain characters the pattern never mentions (digits, punctuation, special tokens). They can never be consumed and die immediately — never treat them as wildcards.

**Why "legal" collapses to "state non-empty" here.** This grammar has no empty alternatives, and by induction on the syntax tree every position can reach some `last` position, so every non-empty state has a completing suffix. Allow empty or unreachable branches and you must add a backward co-accessibility pass, otherwise dead states would be reported as legal.

## Verification

- Reference: recursive-descent parse → Glushkov position automaton (`nullable` / `first` / `follow` / `last`).
- Independent oracle: Thompson NFA with explicit ε transitions + subset construction + ε-closure; the two share no code.
- Third-party witness: brute-force suffixes (length ≤ 4) and confirm with Python `re.fullmatch` that every `mask`-true token really completes, that every `complete`-true token matches directly, and that `accept` equals `re.fullmatch(pattern, prefix)`.
- On 400 random pattern / vocabulary / prefix triples the reference and the oracle agree exactly, and all 400 patterns satisfy "every position is co-accessible".

## Complexity

- Parsing and automaton construction `O(|pattern|)`; each token costs `O(|token|)` transitions; each prefix costs `O(|prefix|)` to replay — `O(|pattern| + |prefix| + Σ|token|)` overall.

## Related problems

- `inf-spec-decoding-rejection`: also picks among candidate tokens, but the criterion is an acceptance rate rather than a grammar.
- `inf-cache-aware-routing`: reusing a "state → precomputed table" mapping is the same class of engineering ledger.
