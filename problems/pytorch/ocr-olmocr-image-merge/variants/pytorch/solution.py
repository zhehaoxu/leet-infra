def solve(images, tolerance=0.5):
    """olmOCR `_merge_image_elements` union-find 容差合并账。

    images: [[name, x0, y0, x1, y1], ...]；
    返回 [{"name": ..., "bbox": [x0, y0, x1, y1]}, ...]，
    输出顺序 = 各连通分量最小原始下标的升序。
    """
    n = len(images)
    parent = list(range(n))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != i:
            parent_i = parent[i]
            parent[i] = root
            i = parent_i
        return root

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def bboxes_overlap(b1, b2):
        h_dist = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        v_dist = max(0, max(b1[2], b2[2]) - min(b1[4], b2[4]))
        return h_dist <= tolerance and v_dist <= tolerance

    for i in range(n):
        for j in range(i + 1, n):
            if bboxes_overlap(images[i], images[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        first = images[indices[0]]
        merged_bbox = [first[1], first[2], first[3], first[4]]
        merged_name = first[0]
        for idx in indices[1:]:
            bbox = images[idx]
            merged_bbox = [
                min(merged_bbox[0], bbox[1]),
                min(merged_bbox[1], bbox[2]),
                max(merged_bbox[2], bbox[3]),
                max(merged_bbox[3], bbox[4]),
            ]
            merged_name += f"+{bbox[0]}"
        merged.append({"name": merged_name, "bbox": merged_bbox})
    return merged
