
def solve(instruction_ids, kwargs_list, response):
    """转写 google-research/instruction-following-eval 的规则判别器（strict 口径，
    词法口径题面钉死；6 类指令）。

    参数
    ----
    instruction_ids : list[str]，与 kwargs_list 等长（≥1），id 必须属于下方 6 类
    kwargs_list     : list[dict]，每类的参数
    response        : str，被评响应

    6 类指令（observed 与 pass 的口径）：
      keywords:existence      kwargs {'keywords': [..]}
        observed = 命中的关键词数；pass = 全部命中（子串、大小写不敏感）
      keywords:forbidden      kwargs {'forbidden_words': [..]}
        observed = 命中数；pass = 命中数为 0
      keywords:frequency      kwargs {'keyword': str, 'frequency': int, 'relation': 'at least'|'less than'}
        observed = 正则 r"\\b<kw>\\b" 在 response.lower() 的匹配数；pass 按关系比较
      length_constraints:number_words  kwargs {'num_words': int, 'relation': 同上}
        observed = len(response.split())（题面钉死空白切分，非 NLTK）；pass 按关系比较
      format:end_phrase       kwargs {'end_phrase': str}
        observed = bool；pass = response.strip().rstrip('"').lower().endswith(end_phrase.lower())
      detectable_format:number_bullet_lists  kwargs {'num_bullets': int}
        observed = 以 "* " 开头的行数；pass = observed >= num_bullets

    返回
    ----
    dict：
      results  list[dict]：{'instruction_id', 'pass' bool, 'observed'} 逐条
      all_pass bool
    """
    raise NotImplementedError
