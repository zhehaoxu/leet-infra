def solve(capacity, ops):
    root = {
        "edge": (),
        "path": (),
        "children": {},
        "parent": None,
        "last": -1,
        "id": 0,
    }
    next_id = [0]
    cached = [0]
    evicted = [0]
    active = {}

    def make_node(edge, path, parent, last):
        next_id[0] += 1
        node = {
            "edge": edge,
            "path": path,
            "children": {},
            "parent": parent,
            "last": last,
            "id": next_id[0],
        }
        parent["children"][edge[0]] = node
        cached[0] += len(edge)
        return node

    def is_pinned(node):
        path = node["path"]
        size = len(path)
        for prompt in active.values():
            if len(prompt) >= size and prompt[:size] == path:
                return True
        return False

    def evict(now):
        while cached[0] > capacity:
            best = None
            stack = [root]
            while stack:
                node = stack.pop()
                if node is not root and not node["children"] and not is_pinned(node):
                    key = (node["last"], node["id"])
                    if best is None or key < best[0]:
                        best = (key, node)
                stack.extend(node["children"].values())
            if best is None:
                return
            node = best[1]
            parent = node["parent"]
            del parent["children"][node["edge"][0]]
            cached[0] -= len(node["edge"])
            evicted[0] += len(node["edge"])

    hits = []
    for now, op in enumerate(ops):
        if op[0] == "finish":
            active.pop(op[1], None)
            evict(now)
            continue

        _, rid, tokens = op
        tokens = tuple(tokens)
        node = root
        pos = 0
        while pos < len(tokens):
            child = node["children"].get(tokens[pos])
            if child is None:
                break
            edge = child["edge"]
            matched = 0
            while (
                matched < len(edge)
                and pos + matched < len(tokens)
                and edge[matched] == tokens[pos + matched]
            ):
                matched += 1
            if matched < len(edge):
                del node["children"][edge[0]]
                mid = make_node(
                    edge[:matched],
                    node["path"] + edge[:matched],
                    node,
                    now,
                )
                child["edge"] = edge[matched:]
                mid["children"][child["edge"][0]] = child
                child["parent"] = mid
                cached[0] -= matched
                node = mid
                pos += matched
                break
            node = child
            node["last"] = now
            pos += matched

        if pos < len(tokens):
            make_node(tokens[pos:], node["path"] + tokens[pos:], node, now)
        hits.append(pos)
        active[rid] = tokens
        evict(now)

    return hits, evicted[0], cached[0]
