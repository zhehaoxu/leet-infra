def solve(logits, blank_id=0):
    """CTC greedy 解码账（PP-OCR BaseRecLabelDecode.decode 口径）。

    输入
    ----
    logits   : [T, V] 数值二维列表（T 帧、V 类，只取 argmax，不做 softmax）
    blank_id : CTC blank 的类 id（PP-OCR 固定 0；题面允许指定，须在 [0, V)

    记账（题面钉死，与官方一致）：
      1. raw_path[t] = argmax(logits[t])，并列取最小类 id；
      2. 相邻去重在**原始序列**上判定（含 blank）：连续相同 id 合并为一个 run；
         因此 `A blank A` 的两个 A 不相邻、都保留——blank 分隔语义；
      3. blank 的 run 整体丢弃；
      4. spans[i] = 输出第 i 个 token 的 run 帧区间 [start, end]（**闭区间**）。

    返回
    ----
    dict：
      raw_path  list[int]     每帧 argmax（原样）
      tokens    list[int]     解码输出 id（collapse + 去 blank）
      spans     list[[int,int]] 每个 token 的 [start,end] 闭帧区间
    """
    raise NotImplementedError
