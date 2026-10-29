# Constrained Decoding: Regex-Grammar Candidate Token Mask

## Problem

Making a model emit only well-formed structure (JSON, SQL, function-call arguments) by asking nicely in the prompt does not work reliably. The engineering answer is **constrained decoding**: before every sampling step, tokens that would drive the grammar into a dead end are masked out. The engine produces a boolean mask and the sampler only picks among the legal positions.

The hard part is not sampling — it is the **state**. A token is not a single character: one token can push the automaton through several states, or kill it on its first character. And "is this token usable right now" asks "can the string still be completed after appending it", not "does appending it already finish the string". Conflating those two questions is the most common bug in this kind of implementation.

This problem asks you to implement the automaton behind a small regex grammar and, for a given vocabulary, answer three things at once: which tokens are still usable, whether the current prefix may already stop, and which tokens finish the string exactly.

## Rules

**Grammar** (the accepted form of `pattern`, lowest to highest precedence):

- `A|B`: alternation, either A or B;
- `AB`: concatenation, adjacent units are joined;
- `X*` / `X+` / `X?`: repeat 0+ / 1+ / 0–1 times;
- `(X)`: grouping;
- `[abc]`: character class, every character inside is a literal (no `^`, no `-`);
- any other character: a literal character.

Empty alternatives (`a|`), empty groups `()`, escapes, `.`, `^`, `$`, `{m,n}` never appear.

Let `L(pattern)` be the set of strings fully matched by `pattern`. Given `pattern`, a candidate vocabulary `vocab`, and the already generated text `prefix`, return a three-tuple, in this order:

- `mask`: a `bool` list of length `len(vocab)`. `mask[i]` is true iff **there exists** a (possibly empty) string `w` such that `prefix + vocab[i] + w` belongs to `L(pattern)`;
- `accept`: a single `bool`, whether `prefix` itself belongs to `L(pattern)` — i.e. "we may stop now";
- `complete`: a `bool` list of length `len(vocab)`. `complete[i]` is true iff `prefix + vocab[i]` belongs to `L(pattern)`. `complete[i]` always implies `mask[i]`.

Mind nullable patterns: for `pattern = "a*"` the empty string is a full match, so `accept` is true for an empty prefix; that does not affect `mask`, which only answers "can we keep going".

## Examples

**Example 1**: `pattern = "a(b|c)*d"`, `prefix = "a"`, `vocab = ["b", "c", "d", "e", "bb"]`

```
"a" + "b"  = "ab"   can still append "d"       -> mask true,  complete false
"a" + "c"  = "ac"   can still append "d"       -> mask true,  complete false
"a" + "d"  = "ad"   already a full match       -> mask true,  complete true
"a" + "e"  = "ae"   'e' is not in the alphabet -> mask false, complete false
"a" + "bb" = "abb"  can still append "d"       -> mask true,  complete false

mask     = [True, True, True, False, True]
accept   = False
complete = [False, False, True, False, False]
```

**Example 2**: `pattern = "ab[cd]?e"`, `prefix = "ab"`, `vocab = ["c", "d", "e", "cde", "ce", "b"]`

```
"abc"   the optional slot is spent on 'c', 'e' still missing -> true  / false
"abd"   same                                                  -> true  / false
"abe"   full match                                            -> true  / true
"abcde" the optional slot holds one char, 'd' is extra        -> false / false
"abce"  full match                                            -> true  / true
"abb"   'b' is not accepted after "ab"                        -> false / false

mask     = [True, True, True, False, True, False]
accept   = False
complete = [False, False, True, False, True, False]
```

**Example 3** (nullable pattern): `pattern = "a*"`, `prefix = "aa"`, `vocab = ["a", "b", "aa"]`

```
mask     = [True, False, True]
accept   = True
complete = [True, False, True]
```

`"b"` can never appear in a match of `a*`, so the second entry of both lists is false; and the prefix itself is already a full match, hence `accept` is true.

## Constraints

- `pattern` is at most 32 characters; `vocab` has 1–64 non-empty tokens of at most 8 characters; `prefix` is at most 32 characters.
- `vocab` and `prefix` may contain characters outside the grammar alphabet; such tokens always get `mask` / `complete` false.
- Only the semantics of the three outputs is judged; any automaton implementation is fine (position automaton, NFA subset construction, memoized derivatives).
- `mask` / `complete` must be `bool` lists and `accept` must be a `bool`.
