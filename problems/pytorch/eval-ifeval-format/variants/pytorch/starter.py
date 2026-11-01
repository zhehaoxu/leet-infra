
def solve(instruction_ids, kwargs_list, response):
    """转写 google-research/instruction-following-eval 的三类结构判别器
    （官方 check_following 原样口径）。

    参数
    ----
    instruction_ids : list[str]，与 kwargs_list 等长（>=1），id 必须属于 3 类白名单
    kwargs_list     : list[dict]，每类参数（json_format/two_responses 传 {}）
    response        : str，被评响应

    3 类指令（observed 与 pass 的口径）：
      detectable_format:json_format   kwargs {}
        strip 后依次剥除前缀 ```json / ```Json / ```JSON / ```（各只一次），
        再剥除后缀 ```（只一次），再 strip；json.loads（捕获 ValueError）
        成功则 pass。observed = bool。
      detectable_content:postscript   kwargs {'postscript_marker': str}
        marker 先 strip；value.lower() 后按 marker 精确分支：
          "P.P.S" -> r"\\s*p\\.\\s?p\\.\\s?s.*$"
          "P.S."  -> r"\\s*p\\.\\s?s\\..*$"
          其他    -> r"\\s*<marker.lower()>.*$"（marker 拼接进正则、
                     不 re.escape，marker 里的 . 等元字符按正则解释）
        re.findall(..., flags=re.MULTILINE)；observed = 匹配条数，
        pass = observed > 0（任意行命中即可）。
      combination:two_responses       kwargs {}
        split("******")；空段位于 index 0 或末段则跳过，中间空段直接 fail；
        pass = 非空段恰 2 段且 strip 后互不相等；observed = 非空段总数。

    返回
    ----
    dict：
      results  list[dict]：{'instruction_id', 'pass' bool, 'observed'} 逐条
      all_pass bool
    """
    raise NotImplementedError
