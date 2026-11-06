"""Qwen3-ForcedAligner NAR 时间戳账——判题用例（order 831）。"""

import torch


def check(solve):
    check_tokenize(solve)
    check_encode(solve)
    check_masked_timestamps(solve)
    check_fix(solve)
    check_parse(solve)
    print("all checks passed")


def check_tokenize(solve):
    """分词：clean_token 保留 L/N/'、CJK 逐字切、空格分段、空段丢弃。"""
    torch.set_default_dtype(torch.float64)
    # "Hello," 去逗号 → "Hello"；"世界！" 去！→ "世界"→CJK 逐字 ["世","界"]；
    # "it's" 保留撇号；"2026" 数字保留；"..." 清洗后为空被丢弃
    text = "Hello, 世界！it's 2026 ..."
    wl = solve(mode="tokenize", text=text)["word_list"]
    assert wl == ["Hello", "世", "界", "it's", "2026"]
    # 段内 CJK 与非 CJK 混排：连续拉丁先合并成一个 token，遇 CJK 切开
    assert solve(mode="tokenize", text="abc中def")["word_list"] == ["abc", "中", "def"]
    # 纯拉丁空格分段 + 标点剔除
    assert solve(mode="tokenize", text="a.b c!!")["word_list"] == ["ab", "c"]
    # 空文本 / 全标点
    assert solve(mode="tokenize", text="")["word_list"] == []
    assert solve(mode="tokenize", text="!!! ???")["word_list"] == []


def check_encode(solve):
    """占位编码：双 <timestamp>、词间与词尾、audio 前缀。"""
    torch.set_default_dtype(torch.float64)
    wl = ["A", "B", "C"]
    text = solve(mode="encode", word_list=wl)["text"]
    ts = "<timestamp>"
    assert text == ("<|audio_start|><|audio_pad|><|audio_end|>"
                    + "A" + ts + ts + "B" + ts + ts + "C" + ts + ts)
    assert text.count(ts) == 2 * len(wl)          # 每词恰 2 个时间戳位
    # 单/空词边界
    assert solve(mode="encode", word_list=["X"])["text"].count(ts) == 2
    assert solve(mode="encode", word_list=[])["text"].count(ts) == 2  # 尾部仍留一对


def check_masked_timestamps(solve):
    """NAR 取值：只在 timestamp_token_id 位置取 argmax 值 × segment_time。"""
    torch.set_default_dtype(torch.float64)
    tid = 7
    # 序列：普通 token(99) 与 tid 交错，仅 tid 位的值有效
    input_ids = torch.tensor([99, tid, 99, tid, tid, 99, tid])
    output_ids = torch.tensor([500, 1, 400, 2, 3, 300, 4])
    res = solve(mode="masked_timestamps", input_ids=input_ids,
                output_ids=output_ids, timestamp_token_id=tid, segment_time=20.0)
    assert res["values"].tolist() == [20.0, 40.0, 60.0, 80.0]
    assert res["count"] == 4
    # 先掩码再乘：非 tid 位的大值（500/400/300）不进入
    assert 500 * 20.0 not in res["values"].tolist()
    # segment_time 标量放大/缩小
    res2 = solve(mode="masked_timestamps", input_ids=input_ids,
                 output_ids=output_ids, timestamp_token_id=tid, segment_time=0.5)
    assert res2["values"].tolist() == [0.5, 1.0, 1.5, 2.0]
    # 全链路一致性：encode 的占位数 == masked 取值数
    wl = ["A", "B"]
    ids = torch.tensor([tid] * (2 * len(wl)))
    vals = torch.arange(2 * len(wl), dtype=torch.float64)
    r = solve(mode="masked_timestamps", input_ids=ids, output_ids=vals,
              timestamp_token_id=tid, segment_time=1.0)
    assert r["count"] == 2 * len(wl)


def check_fix(solve):
    """LIS 单调修复：非降 DP / parent 平分留先 / 段长分治 / int 截断。"""
    torch.set_default_dtype(torch.float64)
    # 例1：单点异常（段长 1），距离平分 (1<=1) 取左
    r = solve(mode="fix", timestamp=torch.tensor([10.0, 20.0, 30.0, 25.0, 40.0, 50.0]))
    assert r["dp"] == [1, 2, 3, 3, 4, 5]
    assert r["parent"] == [-1, 0, 1, 1, 2, 4]
    assert r["lis_indices"] == [0, 1, 2, 4, 5]
    assert r["is_normal"] == [True, True, True, False, True, True]
    assert r["fixed"] == [10, 20, 30, 30, 40, 50]
    # 例2：非降 LIS（允许相等）——全部正常，dp 严格递增
    r2 = solve(mode="fix", timestamp=torch.tensor([5.0, 5.0, 5.0]))
    assert r2["dp"] == [1, 2, 3]
    assert r2["is_normal"] == [True, True, True]
    assert r2["fixed"] == [5, 5, 5]
    # 例3：段长 3 → 线性插值，step 分母 (count+1)=4
    r3 = solve(mode="fix", timestamp=torch.tensor([10.0, 500.0, 400.0, 300.0, 200.0, 600.0]))
    assert r3["is_normal"] == [True, True, False, False, False, True]
    assert r3["fixed"] == [10, 500, 525, 550, 575, 600]
    # 例4：段长 2 距离择一（近左取左、近右取右）
    r4 = solve(mode="fix", timestamp=torch.tensor([10.0, 90.0, 80.0, 20.0, 100.0]))
    assert r4["is_normal"] == [True, True, False, False, True]
    assert r4["fixed"] == [10, 90, 90, 100, 100]
    # 例5：单侧缺失（开头无左 / 结尾无右）
    r5 = solve(mode="fix", timestamp=torch.tensor([500.0, 400.0, 10.0, 20.0]))
    assert r5["fixed"] == [10, 10, 10, 20]
    r6 = solve(mode="fix", timestamp=torch.tensor([10.0, 20.0, 5.0, 3.0]))
    assert r6["fixed"] == [10, 20, 20, 20]
    # 例6：插值浮点 int() 向零截断（12.5→12、17.5→17，非 round）；异常值全 <10 才不进 LIS
    r7 = solve(mode="fix", timestamp=torch.tensor([0.0, 10.0, 9.0, 8.0, 7.0, 20.0]))
    assert r7["is_normal"] == [True, True, False, False, False, True]
    assert r7["fixed"] == [0, 10, 12, 15, 17, 20]
    # 例7：dp.index(max) 平分取最早（idx=3 而非 4；取最后一段链不同，尾部结果不同）
    r8 = solve(mode="fix", timestamp=torch.tensor([10.0, 20.0, 10.0, 20.0, 15.0]))
    assert r8["dp"] == [1, 2, 2, 3, 3]
    assert r8["lis_indices"] == [0, 1, 3]
    assert r8["fixed"] == [10, 20, 20, 20, 20]


def check_parse(solve):
    """词 span：fix 后成对 (2i,2i+1) 映射 + /1000 秒换算。"""
    torch.set_default_dtype(torch.float64)
    wl = ["A", "B", "C"]
    ts = torch.tensor([0.0, 100.0, 1000.0, 2000.0, 3000.0, 4000.0])
    r = solve(mode="parse", word_list=wl, timestamp=ts)
    assert r["fixed"] == [0, 100, 1000, 2000, 3000, 4000]
    assert r["spans"] == [
        {"text": "A", "start_time": 0.0, "end_time": 0.1},
        {"text": "B", "start_time": 1.0, "end_time": 2.0},
        {"text": "C", "start_time": 3.0, "end_time": 4.0},
    ]
    # 非单调输入先 fix 再成对（单点异常取左）
    r2 = solve(mode="parse", word_list=["X", "Y"],
               timestamp=torch.tensor([0.0, 100.0, 90.0, 200.0]))
    assert r2["fixed"] == [0, 100, 100, 200]
    assert r2["spans"] == [
        {"text": "X", "start_time": 0.0, "end_time": 0.1},
        {"text": "Y", "start_time": 0.1, "end_time": 0.2},
    ]
    # 3 位小数（round 到 3 位；fixed 已 int，/1000 恰 3 位）
    r3 = solve(mode="parse", word_list=["Z"],
               timestamp=torch.tensor([1666.0, 2333.0]))
    assert r3["spans"] == [{"text": "Z", "start_time": 1.666, "end_time": 2.333}]
