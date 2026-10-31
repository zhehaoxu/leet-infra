import hashlib
import re

_END_MARKS = (".", "?", "!", '"')
_ELLIPSIS = "..."
_POLICY_SUBSTRINGS = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
]
_CITATION_RE = re.compile(r"\[\d*\]|\[edit\]|\[citation needed\]")


def solve(text, badwords=None, seen_hashes=None,
          min_words_per_line=5, min_num_sentences=3, max_word_length=1000):
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if badwords is not None:
        if not isinstance(badwords, (list, tuple)) or any(not isinstance(b, str) or not b for b in badwords):
            raise ValueError("badwords must be a list of non-empty strings")
    if seen_hashes is not None:
        if isinstance(seen_hashes, (str, bytes)) or not isinstance(seen_hashes, (list, tuple, set, frozenset)):
            raise ValueError("seen_hashes must be a list/set of hash strings")
    for name, v in (("min_words_per_line", min_words_per_line),
                    ("min_num_sentences", min_num_sentences),
                    ("max_word_length", max_word_length)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive int")

    if seen_hashes is None:
        seen = set()
    else:
        seen = set(seen_hashes)

    bw_re = None
    if badwords:
        bw_re = re.compile("[^a-z]({})[^a-z]".format("|".join(badwords)))

    counters = {
        "lines_with_too_long_word": 0,
        "lines_no_endmark": 0,
        "lines_too_short": 0,
        "lines_javascript": 0,
        "lines_policy": 0,
        "lines_dedup": 0,
    }
    valid_lines = []
    new_hashes = []
    reject_reason = None

    for line in text.splitlines():
        line = line.strip()
        if any(len(word) > max_word_length for word in line.split()):
            counters["lines_with_too_long_word"] += 1
            continue
        line = _CITATION_RE.sub("", line)
        if not line.endswith(_END_MARKS) or line.endswith(_ELLIPSIS):
            counters["lines_no_endmark"] += 1
            continue
        if len(line.split()) < min_words_per_line:
            counters["lines_too_short"] += 1
            continue
        line_lower = line.lower()
        if "lorem ipsum" in line_lower:
            reject_reason = "filtered_page_loremipsum"
            break
        if "javascript" in line_lower:
            counters["lines_javascript"] += 1
            continue
        if "{" in line:
            reject_reason = "filtered_page_squigglybracket"
            break
        if any(p in line_lower for p in _POLICY_SUBSTRINGS):
            counters["lines_policy"] += 1
            continue
        if bw_re is not None and bw_re.search(line_lower) is not None:
            reject_reason = "filtered_page_badword"
            break
        h = hashlib.md5(line.strip().lower().encode("utf-8")).hexdigest()
        if h in seen:
            counters["lines_dedup"] += 1
            continue
        seen.add(h)
        new_hashes.append(h)
        valid_lines.append(line)

    if reject_reason is None and len(valid_lines) < min_num_sentences:
        reject_reason = "filtered_page_toofewsentences"
    accepted = reject_reason is None
    num_sentences = len(valid_lines)  # 拒绝前累计的有效行数（题面钉死口径）
    if not accepted:
        valid_lines = []
        new_hashes = []
    return {
        "accepted": accepted,
        "reject_reason": reject_reason,
        "valid_lines": valid_lines,
        "clean_text": "\n".join(valid_lines).strip() if accepted else "",
        "num_sentences": num_sentences,
        "drop_counters": counters,
        "new_hashes": new_hashes,
    }
