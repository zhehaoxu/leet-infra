import torch


def solve(logits, sim, k, lam):
    # TODO: Contrastive Search（CNTRM）候选打分与选择；确定性 argmax，无需随机数入参。
    # ① 候选 = softmax(logits) 降序 top-k（并列按索引升序）→ cand_idx (k,)；
    # ② deg_j = max_i (sim[j,i]+1)/2（T=0 时为 0），夹到 [0,1]；
    # ③ score_j = (1-lam)*log_softmax(logits)[cand_j] - lam*deg_j；
    # ④ chosen = cand_idx[argmax(score)]，并列取候选序里更靠前的。
    # 返回 (cand_idx, deg, scores, chosen)：cand_idx 为 int64 (k,)，deg/scores 为 (k,)，chosen 标量。
    raise NotImplementedError
