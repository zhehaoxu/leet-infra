def solve(stmts):
    """autograd 血缘账：叶 / requires_grad / 区域 / grad_fn 图 / 梯度累加。

    一段程序按顺序给出若干条语句，每条语句要么创建一个新张量，要么就地改动 / 读取已有
    张量。张量用**语句下标**寻址（从 0 开始）。要求逐条给出语句执行后的观测行。
    不许调用 torch 来查表得到答案，请按下面的规则自行推演。

    参数::

        stmts : list，每个元素是一条语句，形如 [kind, *args] 或 [kind, *args, ctx]

    返回值：与 stmts 等长的 list，每个元素是 dict，键固定为::

        {"ok": bool, "rg": bool, "leaf": bool, "infer": bool, "rank": int,
         "node": str, "view_of": int, "grad": bool, "grad_id": int}

    九个观测字段（``ok`` 为 False 时其余八个字段一律为 ``None``，表示这条语句被 torch
    拒绝、程序状态什么都没变）：

    - ``rg``      : ``t.requires_grad``
    - ``leaf``    : ``t.is_leaf``
    - ``infer``   : ``torch.is_inference(t)``
    - ``rank``    : ``t.dim()``
    - ``node``    : ``type(t.grad_fn).__name__``，没有 ``grad_fn`` 时为 ``None``
    - ``view_of`` : 该张量的 ``_base`` 对应的语句下标；不是视图（``_base`` 为
                    ``None``）时为 ``None``
    - ``grad``    : ``t.grad is not None``
    - ``grad_id`` : 当前 ``t.grad`` 缓冲是**哪一条 backward 语句**分配出来的（该语句的
                    下标）；``grad`` 为 False 时为 ``None``

    ``ctx`` 只出现在创建新张量的语句里，取 ``"grad"``（默认求导态）、
    ``"no_grad"``（``torch.no_grad()``）或 ``"inference_mode"``
    （``torch.inference_mode()``）；就地改动与读取类语句一律不带 ``ctx``。

    语句表（``a`` / ``b`` 是语句下标，``flag`` / ``retain`` 是 bool）::

        ["leaf", flag, ctx]            t = torch.ones(2, 3, requires_grad=flag)
        ["add"|"mul", a, b, ctx]       t = ts[a] + ts[b]  /  ts[a] * ts[b]
        ["neg"|"sigmoid"|"relu", a, ctx]
        ["sum"|"mean", a, ctx]         全归约
        ["t"|"index"|"unsqueeze", a, ctx]
                                       t() 转置（仅 rank<=2）/ [0] 取下标（仅 rank>=1）
                                       / unsqueeze(0)
        ["detach"|"clone", a, ctx]
        ["detach_", a]                 就地 detach（视图上非法）
        ["add_", a, b]                 ts[a].add_(ts[b])
        ["requires_grad_", a, flag]    ts[a].requires_grad_(flag)（仅叶合法）
        ["retain_grad", a]             ts[a].retain_grad()
        ["backward", a, retain]        ts[a].backward(retain_graph=retain)
        ["grad", a]                    t = ts[a].grad（没有 .grad 时非法）
        ["grad_none", a]              ts[a].grad = None
        ["grad_zero_", a]             ts[a].grad.zero_()
        ["probe", a]                   重新读取 ts[a] 此刻的状态

    规则：

    1. **叶与 node 互为反面**：``leaf == (node is None)``。有 ``grad_fn`` 就不是叶。
    2. **一元/二元数值运算**（add/mul/neg/sigmoid/relu/sum/mean/clone）：
       ``ctx == "grad"`` 时 ``rg`` 取「所有**非 inference** 操作数里有一个当前 ``rg`` 为真」，
       ``ctx`` 为 ``no_grad`` / ``inference_mode`` 时 ``rg`` 恒为 False；
       ``node`` 仅在 ``rg`` 为真时给出（add→AddBackward0、mul→MulBackward0、
       neg→NegBackward0、sigmoid→SigmoidBackward0、relu→ReluBackward0、
       sum→SumBackward0、mean→MeanBackward0、clone→CloneBackward0），否则为 ``None``；
       ``infer`` 等于「``ctx == "inference_mode"``」；``view_of`` 恒为 ``None``；
       新的 ``grad``/``grad_id`` 为 False/``None``。``rank``：sum/mean 归约为 0，
       其余照抄操作数（add/mul 取两个操作数的较大者）。
       **inference 张量的 ``requires_grad`` 不参与传播**：``inference 张量 + 普通叶``
       结果进图（由普通叶驱动），``inference 张量 + inference 张量`` 结果不进图。
    3. **视图运算**（t/index/unsqueeze）：``infer`` 与 ``rg`` 都照抄操作数
       （**不受 ``ctx`` 影响**：``no_grad`` 里取出的视图仍然 ``requires_grad=True``）。
       ``node`` 仅当 ``ctx == "grad"`` 且操作数当前 ``rg`` 为真时给出
       （t→TBackward0、index→SelectBackward0、unsqueeze→UnsqueezeBackward0）；
       ``view_of`` 指向**根源**张量（操作数自己是视图时取它的 ``view_of``，
       即视图的视图仍报根），若操作数 ``infer`` 为真则为 ``None``。
       ``rank``：t 不变（仅 rank<=2），index 减 1，unsqueeze 加 1。
       **借来的标志**：没有拿到 ``node`` 的视图（基张量当时 ``rg`` 为假，或处在
       ``no_grad``/``inference_mode`` 里）自己**不持有** autograd 元数据，之后每次观测
       它的 ``rg`` 都实时读根源张量当前的标志（根点亮它看着也点亮、根关掉它跟着关），
       ``infer`` 同理。拿到了 ``node`` 的视图自带元数据，``rg`` 冻结为创建时的值。
    4. **detach**：``rg`` 恒为 False，``infer`` 照抄操作数，``view_of`` 恒为 ``None``，
       ``rank`` 不变，``node`` 恒为 ``None``（即使操作数有 ``grad_fn``）。
    5. **leaf 语句**：``rg = flag``、``leaf = True``、``infer = (ctx == "inference_mode")``、
       ``rank = 2``、``node``/``view_of`` 为 ``None``、``grad = False``。
       在 ``inference_mode`` 里创建的叶**可以**带 ``requires_grad=True``；
       ``no_grad`` 不影响叶自己的 ``flag``。
    6. **就地写入 add_**：主体与操作数都不算新张量，行报主体的新状态。
       ``rg`` 取「主体或操作数有一个**当前** ``rg`` 为真」（本类语句恒在求导态）；
       若结果 ``rg`` 为真则 ``node`` 变成 AddBackward0（主体原本是叶也会被拉进图、
       变成非叶），否则 ``node`` 变 ``None``。``view_of`` 与 ``infer`` 与 ``rank``
       保持主体的值不变。
       失败条件：主体**当前** ``rg`` 为真且（主体自己是叶，或主体是「根为叶」的视图）——
       即「requires_grad 的叶、或叶的视图，都不能就地写」。
    7. **detach_**：视图（``view_of`` 非 ``None``）上非法；成功时主体的 ``rg`` 变 False、
       ``node`` 变 ``None``、``leaf`` 变 True、``view_of`` 变 ``None``，其余不动。
       **例外**：``infer`` 为真的张量没有可复位的 autograd 元数据，就地 detach 之后
       ``rg`` 保持原值（inference 张量只有 ``requires_grad_`` 关得掉它的标志）。
    8. **requires_grad_**：先看 ``infer``——``infer`` 为真时置 ``True`` 一律非法
       （哪怕当前已经是 True），置 ``False`` 合法。其余情况：
       ``flag`` 与主体**当前** ``rg`` 相同则是静默空操作（非叶、视图都合法）；
       不同时只在叶上合法（``node`` 非 ``None`` 的张量失败），成功时只改 ``rg``。
    9. **backward**：先判合法性，失败时本行 ``ok=False`` 且不给任何张量分配梯度：
       (a) 主体 ``rank`` 大于 0（隐式梯度只对标量输出合法）；
       (b) 主体**当前** ``rg`` 为 False；
       (c) 主体的 ``node`` 为 ``None`` 且它不持有自己的元数据（借标志的视图）——
           引擎既找不到节点也没有地方累加；
       (d) 主体所在的图里有一个「已被释放」的节点。一次不带 ``retain_graph`` 的反向会
           释放它走过的所有节点，但只有反向时**要回读保存张量**的节点
           （mul→MulBackward0、sigmoid→SigmoidBackward0、relu→ReluBackward0）
           才会因此拦路；add/neg/sum/mean/clone 与三种视图的 backward 不读回保存的数据，
           释放对它们没有影响，同一条图可以再反多次。带 ``retain_graph=True`` 的反向
           不释放任何节点。
       图的边：每条创建 ``node`` 的语句是一个节点，节点的入边是它用到的操作数里
       「当前 ``rg`` 为真、非 inference、且**自己持有元数据**」的那些——操作数是叶则
       记为叶引用，否则记为它当时的节点；借标志的视图能驱动建图却**不留边**；
       ``add_`` 产生的节点额外带一条「主体此前的节点」这条入边。
       从主体的节点出发能到达的节点集合就是这张图。
       成功时：``retain_graph`` 为 False 就释放集合里的 saving 节点。
       梯度落点：集合里的每个叶引用、以及每个调用过 ``retain_grad`` 的节点宿主张量，
       若其 ``.grad`` 当前为空则分配新缓冲（``grad`` 变 True、``grad_id`` 记本条语句下标），
       已有缓冲则**累加进同一个缓冲**（``grad_id`` 不变）。
       主体没有 ``node`` 但持有自己的元数据时（普通标量叶、或被 ``requires_grad_``
       点亮的标量叶）反向直接成功：只有主体自己拿到／累加缓冲。
       ``backward`` 不改主体自身的 ``rg``/``node``（反向结束后 ``grad_fn`` 对象仍然在）。
   10. **grad / grad_none / grad_zero_ / retain_grad / probe**：
       ``grad`` 把 ``ts[a].grad`` 本身作为一个新张量上报（``rg`` False、``leaf`` True、
       ``infer`` False、``rank`` 与 ``a`` 相同、``node``/``view_of`` 为 ``None``、
       ``grad`` False），``a`` 当前没有 ``.grad`` 时非法；
       ``grad_none`` 把 ``a`` 的 ``.grad`` 清空（``grad`` False、``grad_id`` ``None``），
       本身永远合法；
       ``grad_zero_`` 就地清零缓冲，缓冲对象不变（``grad_id`` 不变），没有缓冲时非法；
       ``retain_grad`` 只给宿主张量记上「反向时要把梯度留在我身上」这个标记，当前
       ``rg`` 为假时非法；``probe`` 不改任何状态，只是重新读一行。

    非法输入（抛 ``ValueError``，与上面「语句被 torch 拒绝」是两回事）：
    ``stmts`` 不是列表、语句不是列表或为空、``kind`` 不在语句表里、参数个数不对、
    参数不是 ``int``/``bool``/``str``、``ctx`` 不在三种取值里、就地与读取类语句多给了
    ``ctx``、下标越界或指向「未创建张量的语句 / 失败的语句」、
    ``flag``/``retain`` 不是 bool。空 ``stmts`` 合法，返回 ``[]``。

    程序保证（超出这些范围的 torch 行为不在本题范围内，也不需要在你的实现里处理）：

    - 后续语句只引用**成功创建张量**的语句下标（leaf、数值/视图/detach/clone、grad 这些
      创建类语句；就地与读取类语句的下标不会被引用）。
    - ``add``/``mul``/``add_`` 的两个操作数在 torch 里一定可以广播（``rank`` 相同或其中
      一个为 0，且实际形状匹配），因此这三条语句不会因为形状而失败；本题不需要你推断
      具体形状。``add_`` 的操作数 ``rank`` 与主体相同或为 0。
    - 程序里不会出现「``infer`` 与 ``rg`` 同时为真」的张量作为**操作数**被使用
      （这种张量仍可自己作为主体出现在 detach/clone/视图/就地/读取类语句里）。
    - ``requires_grad_`` 的主体一定自己持有标志（``view_of`` 为 ``None``）：给借标志的
      视图改标志在 torch 里会临时造出一份元数据，本题不覆盖。
    - 反向失败时不会出现「已经给某些张量分配了梯度」的半途而废（要么全成、要么什么都没动）。
    """
    raise NotImplementedError
