from collections import deque


def solve(text, patterns, replacements):
    if any(pattern == "" for pattern in patterns):
        raise ValueError("patterns must be nonempty strings")

    children = [{}]
    failure = [0]
    outputs = [[]]
    for index, pattern in enumerate(patterns):
        state = 0
        for char in pattern:
            if char not in children[state]:
                children[state][char] = len(children)
                children.append({})
                failure.append(0)
                outputs.append([])
            state = children[state][char]
        outputs[state].append(index)

    queue = deque(children[0].values())
    while queue:
        state = queue.popleft()
        for char, child in children[state].items():
            fallback = failure[state]
            while fallback and char not in children[fallback]:
                fallback = failure[fallback]
            failure[child] = children[fallback].get(char, 0)
            outputs[child].extend(outputs[failure[child]])
            queue.append(child)

    all_matches = []
    state = 0
    for end, char in enumerate(text, 1):
        while state and char not in children[state]:
            state = failure[state]
        state = children[state].get(char, 0)
        for index in outputs[state]:
            all_matches.append((end - len(patterns[index]), end, index))
    all_matches.sort()

    selected = []
    consumed = 0
    for match in sorted(all_matches, key=lambda item: (item[0], -item[1], item[2])):
        start, end, index = match
        if start >= consumed:
            selected.append(match)
            consumed = end

    pieces = []
    cursor = 0
    for start, end, index in selected:
        pieces.append(text[cursor:start])
        pieces.append(replacements[index])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), all_matches, selected
