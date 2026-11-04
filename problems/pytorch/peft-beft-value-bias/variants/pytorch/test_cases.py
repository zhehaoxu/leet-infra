"""BEFT -- bypass bias fine-tuning tests."""

"""
EXPECTED values were dumped from the official HF PEFT repo
(src/peft/tuners/beft/layer.py: Linear.forward bypass-bias add,
merge/unmerge state machine, update_layer shape account; paper
arXiv:2509.15974) in a sandbox; anchors were cross-checked against
the explicit base-linear expansion and the merge/unmerge round trip.
Float comparisons use a 1e-9 tolerance.
"""

import numpy as np

EXPECTED = {
 "F1": {
  "in_dim": 4,
  "out_dim": 6,
  "W": [
   [
    0.2995562877502637,
    0.24237779080894106,
    0.28320739441795323,
    -0.23294220898382634
   ],
   [
    0.6711532211923003,
    0.7817673742546976,
    -1.7903080050220619,
    -1.748402801489203
   ],
   [
    0.038435238263354446,
    -0.11824373914203358,
    -1.7445566610842105,
    -0.46946014308389283
   ],
   [
    0.45729973935936413,
    0.5177154906533428,
    -0.2770635268170197,
    -0.664142915806146
   ],
   [
    0.2494118796915765,
    0.2615836907112959,
    -1.5265439234108247,
    -0.33113931395169954
   ],
   [
    -0.6377842272222147,
    0.10719882715742213,
    0.7096149179654692,
    0.30085844543180795
   ]
  ],
  "b_base": [
   0.4679030501729072,
   -0.20493191012544776,
   -0.7409071350000439,
   0.3618396379853795,
   1.9198992741531207,
   -0.22538756319529962
  ],
  "x": [
   [
    [
     -0.9848309379659244,
     0.4573409128131671,
     0.16048920142982762,
     -0.9214988800859957
    ],
    [
     -1.108152357089335,
     -0.18737793842779643,
     0.4428268084752764,
     -0.23210767943346114
    ],
    [
     0.8272179859465014,
     -0.8144713393247613,
     -0.5656502203844694,
     -0.3651179239279022
    ]
   ],
   [
    [
     -0.6187129493373275,
     0.05731628365819024,
     0.9987723890648004,
     1.419073672127401
    ],
    [
     0.9723184024569489,
     2.5161385989502594,
     -0.6481962873525597,
     -0.03349665465339956
    ],
    [
     -1.2968922957206912,
     0.10049062711321136,
     -0.8074523880762338,
     -0.22069702937409055
    ]
   ]
  ],
  "beft": [
   [
    0.16513440805686194,
    1.581867520831043,
    0.4484609765547373,
    0.03302989245546954,
    1.45025762018505,
    -0.6935664431915003
   ]
  ],
  "beft2": [
   [
    0.9966652935978584,
    0.613075936229511,
    0.7763870103074714,
    -0.30291571309906506,
    -1.2753023640826622,
    -0.475751750065748
   ]
  ],
  "base_out": [
   [
    [
     0.543847743700938,
     0.8154559595501832,
     -0.6802125556332195,
     0.7157904307816567,
     1.8540543543667392,
     0.28839329596164887
    ],
    [
     0.27001229523509956,
     -1.4821365561587894,
     -1.4249141208389444,
     -0.21046509373814476,
     0.9953633032241765,
     0.7056927900427206
    ],
    [
     0.4431466859728835,
     1.3645923040874033,
     0.5454104945964865,
     0.7176733059055977,
     2.8975596535684343,
     -1.3515271651629677
    ]
   ],
   [
    [
     0.24875346012434058,
     -4.844597685693923,
     -3.1800783597314197,
     -1.110615231520499,
     -0.2140030841460261,
     1.3110461213372164
    ],
    [
     1.1932520594228675,
     3.6337144303797113,
     0.14548701340413672,
     2.3109626346454624,
     3.8211800185002875,
     -1.0458373040686524
    ],
    [
     -0.07349932875345017,
     0.834670720242703,
     0.7096190142283257,
     0.1910866584726077,
     2.9284186374015135,
     -0.026856460506430285
    ]
   ]
  ],
  "forward": [
   [
    [
     0.7089821517578,
     2.3973234803812264,
     -0.23175157907848215,
     0.7488203232371262,
     3.304311974551789,
     -0.4051731472298514
    ],
    [
     0.4351467032919615,
     0.09973096467225373,
     -0.976453144284207,
     -0.1774352012826752,
     2.4456209234092263,
     0.012126346851220271
    ],
    [
     0.6082810940297454,
     2.9464598249184464,
     0.9938714711512238,
     0.7507031983610672,
     4.347817273753484,
     -2.0450936083544677
    ]
   ],
   [
    [
     0.4138878681812025,
     -3.26273016486288,
     -2.7316173831766823,
     -1.0775853390650294,
     1.2362545360390238,
     0.6174796781457161
    ],
    [
     1.3583864674797295,
     5.215581951210755,
     0.593947989958874,
     2.343992527100932,
     5.271437638685337,
     -1.7394037472601527
    ],
    [
     0.09163507930341178,
     2.416538241073746,
     1.158079990783063,
     0.22411655092807725,
     4.378676257586563,
     -0.7204229036979306
    ]
   ]
  ],
  "forward_zero": [
   [
    [
     0.543847743700938,
     0.8154559595501832,
     -0.6802125556332195,
     0.7157904307816567,
     1.8540543543667392,
     0.28839329596164887
    ],
    [
     0.27001229523509956,
     -1.4821365561587894,
     -1.4249141208389444,
     -0.21046509373814476,
     0.9953633032241765,
     0.7056927900427206
    ],
    [
     0.4431466859728835,
     1.3645923040874033,
     0.5454104945964865,
     0.7176733059055977,
     2.8975596535684343,
     -1.3515271651629677
    ]
   ],
   [
    [
     0.24875346012434058,
     -4.844597685693923,
     -3.1800783597314197,
     -1.110615231520499,
     -0.2140030841460261,
     1.3110461213372164
    ],
    [
     1.1932520594228675,
     3.6337144303797113,
     0.14548701340413672,
     2.3109626346454624,
     3.8211800185002875,
     -1.0458373040686524
    ],
    [
     -0.07349932875345017,
     0.834670720242703,
     0.7096190142283257,
     0.1910866584726077,
     2.9284186374015135,
     -0.026856460506430285
    ]
   ]
  ],
  "forward_multi": [
   [
    [
     1.7056474453556585,
     3.0103994166107375,
     0.5446354312289892,
     0.4459046101380612,
     2.029009610469127,
     -0.8809248972955994
    ],
    [
     1.43181199688982,
     0.7128069009017648,
     -0.20006613397673578,
     -0.48035091438174027,
     1.1703185593265641,
     -0.4636254032145277
    ],
    [
     1.6049463876276038,
     3.5595357611479574,
     1.7702584814586952,
     0.44778748526200224,
     3.072514909670822,
     -2.5208453584202157
    ]
   ],
   [
    [
     1.410553161779061,
     -2.649654228633369,
     -1.955230372869211,
     -1.3805010521640944,
     -0.03904782804363838,
     0.14172792807996815
    ],
    [
     2.355051761077588,
     5.828657887440265,
     1.3703350002663455,
     2.0410768140018667,
     3.9961352746026755,
     -2.2151554973259007
    ],
    [
     1.0883003729012701,
     3.029614177303257,
     1.9344670010905345,
     -0.07879916217098781,
     3.1033738935039015,
     -1.1961746537636786
    ]
   ]
  ],
  "forward_no_base_bias": [
   [
    [
     0.2410791015848928,
     2.602255390506674,
     0.5091555559215617,
     0.38698068525174684,
     1.3844127003986684,
     -0.1797855840345518
    ],
    [
     -0.03275634688094567,
     0.3046628747977014,
     -0.23554600928416308,
     -0.5392748392680548,
     0.5257216492561058,
     0.2375139100465199
    ],
    [
     0.14037804385683825,
     3.1513917350438945,
     1.7347786061512678,
     0.38886356037568787,
     2.4279179996003637,
     -1.8197060451591685
    ]
   ],
   [
    [
     -0.05401518199170466,
     -3.057798254737432,
     -1.9907102481766383,
     -1.4394249770504088,
     -0.6836447381140969,
     0.8428672413410156
    ],
    [
     0.8904834173068223,
     5.420513861336202,
     1.334855124958918,
     1.9821528891155524,
     3.3515383645322165,
     -1.514016184064853
    ],
    [
     -0.37626797086949537,
     2.621470151199194,
     1.898987125783107,
     -0.13772308705730224,
     2.458776983433443,
     -0.495035340502631
    ]
   ]
  ],
  "merged": [
   0.6330374582297691,
   1.3769356107055954,
   -0.2924461584453066,
   0.39486953044084905,
   3.3701568943381703,
   -0.9189540063867999
  ],
  "unmerged": [
   0.4679030501729071,
   -0.20493191012544765,
   -0.7409071350000439,
   0.3618396379853795,
   1.9198992741531204,
   -0.22538756319529962
  ],
  "param_count": 6,
  "layer_setup": {
   "shape": [
    1,
    6
   ],
   "init_zero": True
  }
 }
}

# ---------- shared helpers ----------


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _assert_close_list(got, want, tol=1e-9):
    assert len(got) == len(want), (len(got), len(want))
    for i, (g, w) in enumerate(zip(got, want)):
        assert _close(g, w, tol), (i, g, w)


def _assert_close_2d(got, want, tol=1e-9):
    got = np.asarray(got)
    assert got.shape == (len(want), len(want[0])), (got.shape, len(want), len(want[0]))
    for i in range(len(want)):
        _assert_close_list([float(v) for v in got[i]], want[i], tol)


def _assert_close_3d(got, want, tol=1e-9):
    got = np.asarray(got)
    assert got.ndim == 3 and got.shape[0] == len(want) and got.shape[1] == len(want[0])
    for b in range(len(want)):
        for l in range(len(want[b])):
            _assert_close_list([float(v) for v in got[b, l]], want[b][l], tol)


def _fix():
    f = EXPECTED["F1"]
    return (np.array(f["x"]), np.array(f["W"]), np.array(f["b_base"]),
            np.array(f["beft"]), np.array(f["beft2"]))


# ---------- mode="forward_delta" ----------


def check_forward_delta(solve):
    f = EXPECTED["F1"]
    x, W, b_base, beft, beft2 = _fix()
    got = solve(mode="forward_delta", x=x, weight=W, beft_bias=beft,
                base_bias=b_base)
    _assert_close_3d(got, f["forward"])
    # 互证: 显式展开 x@W.T + b_base + beft
    manual = x @ W.T + b_base + beft.reshape(1, -1)
    _assert_close_3d(got, manual.tolist())
    # 输出维 = out_dim，广播无平面复制
    assert np.asarray(got).shape == (2, 3, f["out_dim"])
    # 零初始化起点（init_weights=True）: delta=0，forward == base
    got0 = solve(mode="forward_delta", x=x, weight=W,
                 beft_bias=np.zeros((1, f["out_dim"])), base_bias=b_base)
    _assert_close_3d(got0, f["forward_zero"])
    _assert_close_3d(got0, f["base_out"])
    # 无 base bias 的层（论文的 bias-free LLM 路径）: base = x@W.T
    got_nb = solve(mode="forward_delta", x=x, weight=W, beft_bias=beft)
    _assert_close_3d(got_nb, f["forward_no_base_bias"])
    # 旁路不改原权重: base_out 锚点独立成立
    _assert_close_3d((x @ W.T + b_base).tolist(), f["base_out"])


# ---------- mode="adapter_sum" ----------


def check_adapter_sum(solve):
    f = EXPECTED["F1"]
    x, W, b_base, beft, beft2 = _fix()
    total = solve(mode="adapter_sum", bias_list=[beft, beft2])
    _assert_close_list([float(v) for v in total],
                       (np.array(f["beft"]) + np.array(f["beft2"])).sum(axis=0).tolist())
    # 多 adapter forward == 先求和再一次加（官方顺序）
    got = solve(mode="forward_delta", x=x, weight=W, beft_bias=total,
                base_bias=b_base)
    _assert_close_3d(got, f["forward_multi"])
    # 空 adapter 列表: 无 bias 注入语义由调用方保证，单 adapter 等价自身
    single = solve(mode="adapter_sum", bias_list=[beft])
    _assert_close_list([float(v) for v in single],
                       np.array(f["beft"]).reshape(-1).tolist())


# ---------- mode="merge_bias" ----------


def check_merge_bias(solve):
    f = EXPECTED["F1"]
    x, W, b_base, beft, beft2 = _fix()
    got = solve(mode="merge_bias", base_bias=b_base, beft_bias=beft)
    _assert_close_list([float(v) for v in got], f["merged"])
    # squeeze 语义: (1, out) -> (out,) 与原 bias 同形相加
    assert np.asarray(got).shape == (f["out_dim"],)
    # merge 后 forward 等价于 merged bias 的普通线性层（bypass 收编）
    got_m = solve(mode="forward_delta", x=x, weight=W,
                  beft_bias=np.zeros((1, f["out_dim"])),
                  base_bias=np.array(f["merged"]))
    _assert_close_3d(got_m, f["forward"])
    # 无 base bias -> ValueError（官方 merge 护栏）
    try:
        solve(mode="merge_bias", base_bias=None, beft_bias=beft)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # safe_merge: NaN 不写回 -> ValueError
    beft_nan = np.array(f["beft"])
    beft_nan[0, 2] = np.nan
    try:
        solve(mode="merge_bias", base_bias=b_base, beft_bias=beft_nan, safe=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # safe 检查对象是「和」: 两个有限数相加溢出 -> 只查操作数发现不了
    big = np.full(f["out_dim"], 1e308)
    try:
        solve(mode="merge_bias", base_bias=big, beft_bias=big, safe=True)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # 非 safe 的 NaN merge 返回 NaN（官方行为: 只有 safe 才拦截）
    got_ns = solve(mode="merge_bias", base_bias=b_base, beft_bias=beft_nan)
    assert not np.isfinite(np.asarray(got_ns)).all()


# ---------- mode="unmerge_bias" ----------


def check_unmerge_bias(solve):
    f = EXPECTED["F1"]
    x, W, b_base, beft, beft2 = _fix()
    merged = np.array(f["merged"])
    got = solve(mode="unmerge_bias", merged_bias=merged, beft_bias=beft)
    _assert_close_list([float(v) for v in got], f["unmerged"])
    # round trip: merge -> unmerge 恢复原 bias
    assert np.allclose(np.asarray(got), b_base, atol=1e-12)
    # 多 adapter merge 后逐个 unmerge 也恢复（加性可逆）
    m2 = solve(mode="merge_bias", base_bias=b_base,
               beft_bias=solve(mode="adapter_sum", bias_list=[beft, beft2]))
    u2 = solve(mode="unmerge_bias", merged_bias=m2, beft_bias=beft)
    u2 = solve(mode="unmerge_bias", merged_bias=u2, beft_bias=beft2)
    assert np.allclose(np.asarray(u2), b_base, atol=1e-12)


# ---------- mode="layer_setup" ----------


def check_layer_setup(solve):
    f = EXPECTED["F1"]
    got = solve(mode="layer_setup", out_features=f["out_dim"])
    assert got["shape"] == f["layer_setup"]["shape"], got
    assert got["param_count"] == f["param_count"]
    assert np.allclose(np.asarray(got["init"]), 0.0)
    # init_weights=False: 只钉 shape 与参数量（随机初始化不出数值锚）
    got_r = solve(mode="layer_setup", out_features=f["out_dim"],
                  init_weights=False)
    assert got_r["shape"] == [1, f["out_dim"]]
    assert got_r["param_count"] == f["out_dim"]
    assert "init" not in got_r
    # 可训练参数只有旁路 bias: 每层 out 个（无权重矩阵增量）
    assert got["param_count"] == f["out_dim"] == f["layer_setup"]["shape"][1]


# ---------- solve dispatch ----------


def check_dispatch(solve):
    try:
        solve(mode="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


CHECKS = [
    check_forward_delta,
    check_adapter_sum,
    check_merge_bias,
    check_unmerge_bias,
    check_layer_setup,
    check_dispatch,
]
