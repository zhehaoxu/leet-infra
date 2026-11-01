def solve(matches, models):
    """MT-Bench 式 LLM-as-judge pairwise 评测的位置偏差账。

    背景机制（FastChat fastchat/llm_judge，MT-Bench 论文的官方实现）：
      裁判 LLM 对一对回答做 pairwise 判决时存在位置偏差——偏向放在 A 位
      的回答。官方对策是每个问题打两局：g1 以 (answer_1, answer_2) 摆位、
      g2 交换摆位；两局结论一致才算真胜负，不一致（含平局）按平局计。
      裁判的判决文本用 "[[A]]" 格式解析，多场比赛再聚合成胜率。

    输入：
      matches : list[dict]，每场含
                "question_id"（任意标量）、
                "model_1"/"model_2"（模型名字符串）、
                "g1_judgment"/"g2_judgment"（裁判原始输出文本，
                g1 是 (answer_1, answer_2) 摆位的判决、g2 是交换摆位的判决）。
      models  : list[str]，需要给出统计的全部模型名。

    判决解析（output_format "[[A]]"，官方实现为顺序包含判断）：
      对每局裁判文本，按顺序检查：
        文本包含 "[[A]]" → 该局 A 方胜；
        否则包含 "[[B]]" → B 方胜；
        否则包含 "[[C]]" → 平局；
        否则 → error（解析失败）。
      匹配大小写敏感（"[[a]]" 不算）。先命中先判——同一文本里同时出现
      多个标记时以最先满足的为准（官方 if/elif 链的顺序 quirk）。

    摆位映射（swap 账）：
      g1 的 A/B 对应 (model_1/model_2)；g2 因为交换了摆位，A 对应
      model_2、B 对应 model_1。tie/error 不映射，原样保留。

    场次结算（官方 show_result 的聚合条件，逐场）：
      - g1 或 g2 任一局为 error → 整场剔除（outcome = "error"，
        不计入任何模型的统计）。
      - g1 为 tie，或 g1 与 g2 映射后不一致 → 平局（outcome = "tie"，
        两个模型各记 tie 一次）。位置偏差导致的"两局各判一方"正落在此。
      - 两局一致且非 tie → 胜负分明（outcome = 胜者模型名，
        胜者记 win、负者记 loss）。

    统计与胜率：
      每个模型统计 win/loss/tie；胜率
        win_rate_adjusted = (win + 0.5 * tie) / (win + loss + tie)
      （每场平局算半胜半负）。分母为 0 的模型 win_rate_adjusted = 0.0。
      stats 必须覆盖 models 列出的全部模型（没出现的模型全 0 行）。

    返回 dict：
      {
        "matches": [  # 与输入等长、同序
          {"question_id": ..., "g1_winner": ..., "g2_winner": ...,
           "outcome": ...}   # g1_winner/g2_winner 为映射后的
                             # "model_1"/"model_2"/"tie"/"error"
        ],
        "stats": {model: {"win": int, "loss": int, "tie": int,
                          "win_rate_adjusted": float}}
      }
    """
    # TODO
    raise NotImplementedError
