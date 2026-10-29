def solve(documents, vocab, ignored_tokens):
    index = {token: position for position, token in enumerate(vocab)}
    if len(index) != len(vocab):
        raise ValueError("vocab must contain unique tokens")

    ignored = set(ignored_tokens)
    vocab_counts = [0] * len(vocab)
    vocab_doc_freq = [0] * len(vocab)
    oov_counts_by_token = {}
    oov_doc_freq_by_token = {}
    document_stats = []
    for document in documents:
        kept = 0
        oov = 0
        seen = set()
        for token in document:
            if token in ignored:
                continue
            kept += 1
            seen.add(token)
            if token in index:
                vocab_counts[index[token]] += 1
            else:
                oov += 1
                oov_counts_by_token[token] = oov_counts_by_token.get(token, 0) + 1
        for token in seen:
            if token in index:
                vocab_doc_freq[index[token]] += 1
            else:
                oov_doc_freq_by_token[token] = oov_doc_freq_by_token.get(token, 0) + 1
        document_stats.append((kept, oov))

    oov_tokens = sorted(oov_counts_by_token, key=lambda token: (-oov_counts_by_token[token], token))
    known_count = sum(vocab_counts)
    total_count = known_count + sum(oov_counts_by_token.values())
    known_types = sum(count > 0 for count in vocab_counts)
    total_types = known_types + len(oov_tokens)
    return {
        "vocab_counts": vocab_counts,
        "vocab_doc_freq": vocab_doc_freq,
        "oov_tokens": oov_tokens,
        "oov_counts": [oov_counts_by_token[token] for token in oov_tokens],
        "oov_doc_freq": [oov_doc_freq_by_token[token] for token in oov_tokens],
        "document_stats": document_stats,
        "token_coverage": known_count / total_count if total_count else 0.0,
        "type_coverage": known_types / total_types if total_types else 0.0,
    }
