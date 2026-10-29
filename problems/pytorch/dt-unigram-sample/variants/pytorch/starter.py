def solve(text, vocab, log_scores, alpha, uniforms):
    """按显式 uniform 协议采样完整 Unigram 切分；仅用标准库。

    中文契约：
    - text 为长度 0..12 的 Unicode 字符串，位置按 Python 码点计。
      vocab 是 0..20 个字符串的列表，合法项长度 1..12；空项或重复项必须
      抛 ValueError，即使 text 为空也先验证。除这两类非法项外保证合法。
      log_scores 为等长列表，项为有限 Python float，范围 [-100, 0]。
      alpha 为有限数，范围 [0, 4]。uniforms 是恰长 len(text) 的列表，
      项为 Python float 且 0 <= u < 1。其余输入保证合法。
    - 完整路径是按顺序拼接后恰等于 text 的 vocab 下标序列，允许重复使用项。
      路径分数 S 是所选 log_scores 之和，权重为 exp(alpha*S)。Z 为所有
      完整路径权重之和。没有完整路径抛 ValueError。必须稳定计算 logZ，
      不能直接对极小的整路径权重求和。alpha=0 时对完整路径均匀，非局部均匀。
    - 前缀递推：L[0]=0；L[t]=logsumexp(L[s]+alpha*log_scores[i])，
      候选满足 s=t-len(vocab[i])>=0、text[s:t]==vocab[i] 且 s 可达。
      空候选 L[t]=-inf。从 t=len(text) 开始向左采样，候选按 vocab 下标升序。
      每步依次消费 uniforms 中一个值，单候选也消费；概率为
      exp(L[s]+alpha*log_scores[i]-L[t])。选择第一个累计概率严格大于 u
      的候选（等号落下一项）；浮点累计尾误差可兜底选择最后候选。
      正概率不得因下溢被排除；u=0 必须选第一个可达候选，可用对数 CDF。
      跳到所选 s，直到 0，最后将所选下标逆序为从左至右的 ids。
    - 返回四元 tuple：(ids, log_partition, log_probability, uniforms_used)。
      ids 是 Python int 列表；其余三项类型依次为 float、float、Python int，
      log_partition=logZ，log_probability=alpha*S-logZ，
      uniforms_used=采样步数=len(ids)，未使用 uniforms 尾部忽略。
      空 text 在验证 vocab 后返回 ([], 0.0, 0.0, 0)。
    - 不修改任何输入，不读全局 RNG；既非 Viterbi，也非单次整路径 CDF。
      本题不承诺 SentencePiece Sample 的算法或种子级复现。

    English contract:
    - text contains 0..12 Unicode code points (Python string indexing).
      vocab is a list of 0..20 strings, each valid item of length 1..12.
      Empty or duplicate items raise ValueError, including for empty text.
      log_scores is an equally sized list of finite Python floats in [-100, 0].
      alpha is finite in [0, 4]. uniforms is a list of exactly len(text) Python
      floats in [0, 1). All inputs other than the specified invalid vocabulary
      items and unsegmentable text are guaranteed valid.
    - Each complete path is a sequence of vocabulary IDs whose strings
      concatenate to text; IDs may recur. With S the sum of its log scores,
      the path has weight exp(alpha*S). Z sums weights over all complete paths.
      No complete path raises ValueError. Compute logZ stably; alpha=0 is
      uniform over complete paths, not over local choices.
    - L[0]=0. L[t]=logsumexp(L[s]+alpha*log_scores[i]) for reachable s>=0
      with s=t-len(vocab[i]) and text[s:t]==vocab[i]; no candidates means -inf.
      Sample backward from t=len(text). Order eligible ending tokens by
      ascending vocabulary ID. At every step consume the next uniform, even
      with one candidate. Its probability is exp(L[s]+alpha*log_scores[i]-L[t]).
      Choose the first cumulative probability strictly greater than u; exact
      equality belongs to the next candidate. The final candidate may absorb
      floating-point cumulative tail error. Never drop positive mass through
      underflow: u=0 selects the first reachable candidate; a log CDF is suitable.
      Set t=s; stop at 0 and reverse IDs.
    - Return exactly (ids, log_partition, log_probability, uniforms_used), a
      tuple with types list[Python int], float, float, Python int. The floats
      are logZ and alpha*S-logZ. uniforms_used=len(ids); ignore unused uniforms.
      After validating vocab, empty text returns ([], 0.0, 0.0, 0).
    - Do not mutate inputs or consult global RNG. This is neither Viterbi nor
      one whole-path CDF draw, and makes no SentencePiece seed-parity claim.
    """
    raise NotImplementedError
