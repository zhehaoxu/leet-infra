"""STaR（ezelikman/STaR, arXiv:2203.14465）自举理性化账 -- 你的实现。

官方 device_inference.py 把「生成 -> 抽取答案 -> 判对错 -> 答对样本写回
训练集」组织成 eval_output / eval_examples 两个确定性账本：三种数据集
（cqa / gsm / arithmetic）各有一条钉死的答案抽取链，判对的输出按抽取
用中间串拼回 base context 组装新训练样本；eval_examples 再按
eval_batch_size 分批，答错样本以概率 p_rationalization 进 hint 池，
池满一批就以 key+'_r' 重命名、用带提示上下文重评（理性化）。本题把
LLM 生成全部显式入参，请实现下面的 solve；不允许 import numpy /
torch / transformers。

参数
----
mode : str
    "extract"（默认）| "score" | "context" | "eval"。
outputs : list[str]
    mode="extract" 用：每个样本已解码的模型输出原文。
answers : list[str]
    与 outputs 对齐的标准答案（比较两侧 .lower()）。
base_contexts : list[str]
    与 outputs 对齐的保存用 base context 前缀。
classes : list[str]
    与 outputs 对齐的 accuracy 分桶键（cqa/gsm 恒为数据集名，
    arithmetic 为位数串）。
dataset_mode : str
    "cqa" | "gsm" | "arithmetic"。
accuracy : dict | None
    传入则在其上累计（{'class': {'accurate': int, 'total': int}}）。
direct, few_shot_train : bool
    arithmetic direct 抽取取下标 1（行数>1）；few_shot_train 时
    arithmetic 判对样本 raise NotImplementedError。
endoftext : str
    保存时拼接的终止符（抽取链内的判断硬编码 '<|endoftext|>'）。
counts : dict
    mode="score" 用：{'accurate': int, 'total': int}。
example : dict | str
    mode="context" 用：cqa/gsm 为 dict、arithmetic 为原文串。
hint, direct : bool
    mode="context" 用：hint 在 cqa 标 (CORRECT)、gsm 追加 ' (answer)'、
    arithmetic 追加 ' ' + 下一段倒数第 5 行。
dataset : list[(key, example)]
    mode="eval" 用：(分桶键, 样本) 对列表。
prompts, hint_prompts, train_prompts : list[str] | None
    few-shot 行列表；prompts=None 表示 no_prompt（输入无前缀）；
    train_prompts 是 few_shot_train 保存时的前缀行。
gen_fn : callable
    gen_fn(contexts: list[str], hint: bool) -> list[str]，确定性替身。
rng : random.Random
    翻硬币用（p_show_hint_save / p_rationalization）。
eval_batch_size : int
    分批大小（>=1）。
rationalize : bool
    是否启用 hint 池 flush。
p_rationalization, p_show_hint_save : float
    [0,1] 准入/保存概率。

返回
----
mode="extract"：{"successful": list[int], "train_examples": list[str],
    "accuracy": dict}。抽取链（官方钉死，except IndexError 静默跳过，
    total 在 try 前自增）：
      cqa：首行 -> 剪 '<|endoftext|>' 子串 -> 预测 = 首行[-3] 字符；
      gsm：扫到首条含 '####' 的行截断到该行（含）-> 剪 endoftext ->
          预测 = split('####')[-1].strip()；无 '####' 则预测空串；
      arithmetic：行列表上 '<|endoftext|>' / '</scratch>' 为整行相等
          成员测试；endoftext 行下标-1（下标 0 -> -1 取最后一行）；
          '</scratch>' 行下标+1 == len 则跳过；否则 direct 且行数>1
          取 1、否则 0；保存 join(行[:预测行+1]) 再剪 endoftext；
      保存样本 = base_context + 抽取中间串 + endoftext；判对才保存。
mode="score"：total==0 -> int 0，否则 accurate/total。
mode="context"：组装好的上下文字符串（cqa/gsm/arithmetic 三式）。
mode="eval"：{"accuracy", "train_examples"}。每凑满 eval_batch_size
    评一批（不足最后一批永不评测）；批内逐样本：hintless/base 两份
    上下文，arithmetic 时 few-shot 改取 base.split('\\n\\n')[:-1]、两份
    取 split('\\n\\n')[-1]，rng.random() < p_show_hint_save 则保存用
    hintless 换成 base，few_shot_train 时保存加 train_prompts 前缀
    （arithmetic 直接 NotImplementedError）；输入上下文 = few-shot
    join 前缀（prompts=None 则空）+ base；target：arithmetic 取倒数
    第三行（行数<3 为 'invalid'）、cqa 取 answerKey、gsm 取
    answer.split('#### ')[-1]；批评完答错样本以 rng.random() <
    p_rationalization 进 hint 池；rationalize 且池满一批立即取前一批
    （key+'_r' -> accuracy 独立桶）以 hint 上下文重评，剩余留池。

异常
----
mode 未注册、dataset_mode 非法、列表长度不齐、类型不合法 ->
ValueError；arithmetic + few_shot_train + 判对 -> NotImplementedError
（官方 raise，不被 except IndexError 吞）。
"""


def solve(mode="extract", outputs=None, answers=None, base_contexts=None,
          classes=None, dataset_mode=None, accuracy=None, direct=False,
          few_shot_train=False, endoftext="<|endoftext|>", counts=None,
          example=None, hint=False, dataset=None, prompts=None,
          hint_prompts=None, gen_fn=None, rng=None, eval_batch_size=None,
          rationalize=False, p_rationalization=1.0, p_show_hint_save=0.0,
          train_prompts=None):
    raise NotImplementedError("TODO: 实现 extract / score / context / eval 四入口")
