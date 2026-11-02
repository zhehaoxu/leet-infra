"""
Self-RAG 反思 token 打分账。

Self-RAG（arXiv 2310.11511）推理期把「检索触发」与「证据重排」都压
在反思 token 的 log 概率上：给定无检索生成的反思 token logprob、
每条证据的生成 token 序列与逐位置 logprob、以及聚合开关，复算
do_retrieve 判定、segment 级 relevance/ground/utility 加权分与最终
答案挑选。回源 AkariAsai/self-rag main：
retrieval_lm/run_short_form.py::call_model_rerank_w_scores_batch、
postprocess_answer_option_conditioned，retrieval_lm/utils.py 的
反思 token 名单与 postprocess。

solve(no_retr, evidences, second_text, mode="adaptive_retrieval",
      threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5,
      use_seqscore=False, use_groundness=False, use_utility=False,
      closed=False)

no_retr：无检索那次生成 {"text": str, "lp0": {token: logprob}}；
evidences：[{"text", "tokens", "lps", "cumulative_logprob"}]（tokens
与 lps 等长，lps[i] 为第 i 个位置的 logprob 表）；second_text：
do_retrieve 为 False 时第二次生成（追加 "[No Retrieval]"）的文本。

返回账：{"do_retrieve", "retrieval_ratio", "result_keys",
"evidence_scores", "answer", "best_path", "answer2score",
"answer_score", "n_evidence"}。

域约束（ValueError）：threshold 路径下 lp0 缺反思 token（官方先赋
-100 死代码、随即 KeyError）；tokens 与 lps 长度不一致或为空（官方
lps[0] 越界）；do_retrieve 为 True 却没有证据（官方 UnboundLocalError
/ 聚合空列表 IndexError）。
"""


def solve(no_retr, evidences, second_text, mode="adaptive_retrieval",
          threshold=None, w_rel=1.0, w_sup=1.0, w_use=0.5,
          use_seqscore=False, use_groundness=False, use_utility=False,
          closed=False):
    raise NotImplementedError
