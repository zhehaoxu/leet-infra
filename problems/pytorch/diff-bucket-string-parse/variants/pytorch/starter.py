def solve(buckets_str):
    """解析分辨率分桶字符串，返回按输入顺序排列的 [(height, width), ...]。

    buckets_str 形如 "1024,1024; 768,1344"：用 ";" 分隔若干 "高度,宽度" 对。

    - 非字符串、空串或只含空白的字符串抛 ValueError（空白串在切分后得不到合法桶）；
    - 先整体 strip 再按 ";" 切分，每一段必须完整匹配 ^\\s*(\\d+)\\s*,\\s*(\\d+)\\s*$：
      只认十进制数字字符，逗号两侧与段首尾可有空白，多一个逗号、负号、加号、小数点、
      科学计数法或 "512x768" 这类分隔符都抛 ValueError；
    - 两个数字转成 int 后必须为正（0 非法，抛 ValueError）；
    - 任一边不能被 8 整除时只发一条 warning（消息里包含 "not divisible by 8"），桶仍然保留；
    - 结果保持输入顺序，重复桶不合并，元素是 (int, int) 元组。
    """
    # TODO
    raise NotImplementedError
