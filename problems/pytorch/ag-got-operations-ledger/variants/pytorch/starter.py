"""Graph of Thoughts (GoT) 操作账。

把官方 operations.py 的 Thought/Operation 家族在线性操作链上无状态化。

ops 为操作描述符列表，按序组成一条链；每个操作从唯一前驱取 thoughts，
首个操作无前驱（此时多数操作 ValueError，generate/selector 可用
base_state 作种子）。

支持的操作：
- {"op": "score", "combined": bool, "scores": [...]}
    combined=True 与 thoughts zip（scores 更短则截断）；
    combined=False 逐 thought 赋分。赋分置 scored=True。
- {"op": "validate_and_improve", "improve": bool, "num_tries": int,
   "valids": [...], "updates": [...]}
    每轮验证消费一个 valid（置 validated=True）；improve 且无效且
    尝试次数 < num_tries 时消费一个 update 做 {**state, **update}。
    恒定无效时验证恰 num_tries+1 次。改进产物是全新缺省 thought
    （score 复位 0.0、全部标志复位）。
- {"op": "generate", "num_branches_prompt": p, "num_branches_response": r,
   "parsed": [[state, ...], ...]}
    parsed 逐父槽位消费，新 state = {**父state, **ns}；
    前驱有前驱但零 thoughts → 本操作直接跳过。
- {"op": "improve", "updates": [...]}：{**state, **update} 全新 thought。
- {"op": "aggregate", "parsed": dict 或 [dict, ...]}
    base_state 按前驱 score 升序（stable）合并——score 高者后合并、
    覆盖同名键；未打分缺省 0.0。
- {"op": "keep_best_n", "n": int, "higher_is_better": bool}
    全部前驱须已打分；按 score 排序（stable 平分保原序）取前 n。
    排序失败（如含已打分的 score=None）走 fallback：只保留
    isinstance(score, float) 的 thought 再取前 n。n<=0 ValueError。
- {"op": "keep_valid"}：保留 not validated or valid 的 thought
    （未验证的放行）。
- {"op": "ground_truth", "results": [...], "raises": [...]}
    raises[i] 为真 → solved=False（模拟求值异常）；solved setter
    总置 compared_to_ground_truth=True。
- {"op": "selector", "indices": [...]}：按索引选取。

solve(ops, base_state=None) 返回 {"steps": [...], "final": [...]}；
每步记录 {"op", "count", "thoughts"} 加操作特有键：
score→scores；validate_and_improve→attempts（每输入 thought 的
验证次数）；generate→skipped；aggregate→base_state；
keep_best_n→fallback；ground_truth→solved_count。

thought 记账键：state/score/scored/valid/validated/solved/
compared_to_ground_truth；缺省 score=0.0，其余布尔 False。
"""
