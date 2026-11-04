"""AdaMSS -- adaptive multi-subspace scatter adaptation tests."""

"""
EXPECTED values were dumped from the official HF PEFT repo
(src/peft/tuners/adamss/layer.py: update_layer scatter_index account,
forward adapter path, get_delta_weight / get_delta_bias merge trunk)
in a sandbox; anchors were cross-checked against three independent
accounts -- the forward chain x_aug @ newB.T -> per-subspace A_i/B_i,
the merged weight path cat([B_i @ A_i]) @ newB scattered to (out, in+1),
and the equivalence x @ W_delta.T (+ b_delta) == forward with bias.
KMeans/SVD/PCA/QR are all done offline; seg/A_i/B_i/newB are explicit
inputs. Float comparisons use a 1e-9 tolerance.
"""

import numpy as np

EXPECTED = {
 "F1": {
  "r": 5,
  "in_dim": 5,
  "out_dim": 6,
  "subspace_rank": 2,
  "seg": {
   "0": [
    0,
    1
   ],
   "1": [
    2,
    3
   ],
   "2": [
    4
   ]
  },
  "train_idx": [
   0,
   1,
   2
  ],
  "rank_actual": [
   2,
   2,
   1
  ],
  "scatter_index": [
   0,
   1,
   2,
   3,
   4
  ],
  "scatter_index_partial": [
   0,
   1,
   4
  ],
  "newB": [
   [
    0.936650038779724,
    0.01804985703847165,
    0.6927666696591163,
    0.3154090546513482,
    1.5041856647552274
   ],
   [
    -2.0066907365561706,
    -2.1310340831936396,
    -0.19843957784596147,
    0.6365395187683154,
    -0.410549281758414
   ],
   [
    0.42578068000505975,
    -1.173544797067956,
    -1.2593832343358193,
    -0.7165758868835466,
    0.8413227181749481
   ],
   [
    0.696358369654102,
    -0.713252712154986,
    0.18111936994597383,
    -0.6475708354871698,
    0.020059810497343504
   ],
   [
    -1.1142963805907677,
    -0.04655434751901477,
    -0.735753934980961,
    -0.04435950785702231,
    -0.6520426060122302
   ]
  ],
  "newB_b": [
   [
    -1.6497062609235813,
    -1.0422629538107075,
    -0.723476319045549,
    0.6555863140819842,
    0.3020801614586051,
    0.2829882917455624
   ],
   [
    0.17293243462011323,
    0.6364764540263609,
    0.9453478418566444,
    -0.55471618453936,
    -1.5909121646919766,
    0.6673244554529724
   ],
   [
    -0.20942920166475995,
    0.04880537673064059,
    -0.1738052649063213,
    -1.1125682987307608,
    -1.5106513146782443,
    0.9642801867742248
   ],
   [
    1.0930925298253509,
    0.9312396637512143,
    0.20823464575150918,
    0.2685207840712548,
    -0.4189296731440312,
    0.068012235199191
   ],
   [
    -0.402836079808958,
    -0.5436861645760849,
    0.20520316656139895,
    1.698100442046864,
    -1.8835414686235434,
    0.601511318239193
   ]
  ],
  "A": [
   [
    [
     -0.39530128858657,
     0.2639148850157296,
     0.6071282687955677,
     -0.9721597997272025,
     0.7676642531398922
    ],
    [
     0.25505812177433956,
     0.7829798706300901,
     0.27241823407236765,
     1.1620081255305676,
     -0.9377563787668719
    ]
   ],
   [
    [
     1.776099733817743,
     1.2023898441352874,
     -0.5999578484570015,
     0.6601663130603562,
     0.44476738448754977
    ],
    [
     -1.7457635656172505,
     0.5952162456256633,
     -0.5853991209052885,
     -0.25001353358513434,
     -0.6023864391110364
    ]
   ],
   [
    [
     -0.4280840067640782,
     0.07173470176248886,
     0.09671912405619901,
     -1.5591518413922796,
     -0.26870603364939344
    ]
   ]
  ],
  "B": [
   [
    [
     -1.3448111474077982,
     -1.2707778590591046
    ],
    [
     -0.346953600730839,
     0.85530027587691
    ]
   ],
   [
    [
     0.6308565040330028,
     -0.6046580474158655
    ],
    [
     -0.7104208239983351,
     -0.8274098970826956
    ]
   ],
   [
    [
     0.14274021925565974
    ]
   ]
  ],
  "x": [
   [
    [
     -0.8217414802839252,
     0.11343128787139657,
     -0.33779536082531136,
     0.021671175782686587,
     -0.419085181552037
    ],
    [
     0.6137428973723346,
     -0.8156847530495072,
     0.7687510449895725,
     1.331974309444846,
     -0.26888770649361554
    ],
    [
     1.7452430767280316,
     2.029698196865837,
     0.4430750030154757,
     -1.3576051491193484,
     -1.6156939484528159
    ]
   ],
   [
    [
     -0.1757786841969012,
     1.7759298264822208,
     0.574549849965791,
     0.7580309194582652,
     -0.2509133097904481
    ],
    [
     -0.7441468728010533,
     0.37192110436980064,
     -1.1087787163755936,
     -0.19694438865759697,
     0.511152295497356
    ],
    [
     0.03593415340311622,
     0.5245642290612524,
     -0.4459765954576025,
     0.5981798358088446,
     0.4046503807009621
    ]
   ]
  ],
  "forward": [
   [
    [
     -1.7305567019543806,
     -2.1370593389319414,
     -2.3596241320868416,
     -2.4659740788732467,
     0.2195017153483142,
     0.0
    ],
    [
     -0.6819833850275283,
     2.675266635411234,
     2.410759128551939,
     -2.6853036365594174,
     -0.08754099093457607,
     0.0
    ],
    [
     13.460193625793401,
     -2.800933482351035,
     -5.500214955119567,
     8.006553568091265,
     -0.17157183235632564,
     0.0
    ]
   ],
   [
    [
     8.523804903592204,
     -3.926425642865667,
     -1.9410217403810737,
     1.1322744133013056,
     0.3149537486785782,
     0.0
    ],
    [
     -2.007172256532387,
     -2.4205209192723474,
     -1.0722708277273612,
     0.6395037426242052,
     0.20709929411913747,
     0.0
    ],
    [
     1.5876228298798574,
     -1.3752417838346618,
     0.3296462210399587,
     1.38311984443875,
     0.13779340510215832,
     0.0
    ]
   ]
  ],
  "forward_b": [
   [
    [
     -2.6570960857354615,
     -1.3889844899721362,
     4.6244705712820835,
     1.151298040093572,
     0.007681494532781496,
     0.0
    ],
    [
     0.14029027834805924,
     -2.662218171617201,
     3.8001710966671896,
     -0.6939328150046696,
     -0.29914286987998684,
     0.0
    ],
    [
     -15.53254460405876,
     7.381894113168906,
     -9.486621220316023,
     -6.2805811032274725,
     -0.419681120663215,
     0.0
    ]
   ],
   [
    [
     -4.041882427318821,
     1.796456677183427,
     0.26373524336316256,
     -3.062573182990413,
     -0.38810587713669953,
     0.0
    ],
    [
     0.8390764572255203,
     -0.16616971327061383,
     2.794255742537916,
     2.464586338061189,
     0.11284875596237075,
     0.0
    ],
    [
     0.87530934730105,
     0.11442148284394915,
     1.548054225598337,
     0.3764151997828964,
     -0.16613528780253367,
     0.0
    ]
   ]
  ],
  "forward_partial": [
   [
    [
     -1.7305567019543806,
     -2.1370593389319414,
     0.0,
     0.0,
     0.2195017153483142,
     0.0
    ],
    [
     -0.6819833850275283,
     2.675266635411234,
     0.0,
     0.0,
     -0.08754099093457607,
     0.0
    ],
    [
     13.460193625793401,
     -2.800933482351035,
     0.0,
     0.0,
     -0.17157183235632564,
     0.0
    ]
   ],
   [
    [
     8.523804903592204,
     -3.926425642865667,
     0.0,
     0.0,
     0.3149537486785782,
     0.0
    ],
    [
     -2.007172256532387,
     -2.4205209192723474,
     0.0,
     0.0,
     0.20709929411913747,
     0.0
    ],
    [
     1.5876228298798574,
     -1.3752417838346618,
     0.0,
     0.0,
     0.13779340510215832,
     0.0
    ]
   ]
  ],
  "dW": [
   [
    2.112745638322741,
    4.358200871476191,
    1.7279641232347078,
    0.14185824172794076,
    -0.21914809289842704
   ],
   [
    1.2996974203659213,
    -2.15152130457065,
    1.130451848843385,
    -0.35058515516938815,
    1.0392523067170458
   ],
   [
    0.9045869095273935,
    -1.2163389905959114,
    1.0876551405278154,
    0.5614710986939586,
    2.6798359271121033
   ],
   [
    2.8753732849514093,
    1.9829984150230433,
    -0.9346991714018913,
    -1.2904508958862524,
    1.4695487365451008
   ],
   [
    -0.18414107808213095,
    0.12139749316708905,
    -0.07383878123140582,
    0.12317263869373725,
    -0.06395689181074668
   ],
   [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ]
  ],
  "dW_full": [
   [
    -0.5814558893030403,
    -1.376442629123267,
    -1.22672558106904,
    2.4034534576646855,
    3.7374694121460803,
    -1.8789206743515927
   ],
   [
    1.394623035456346,
    1.8192374161464697,
    0.3435363096799013,
    -1.5694481371585685,
    0.6085544865739957,
    -0.04423040979824179
   ],
   [
    -3.1550494099050836,
    -1.8374659884334366,
    -0.9426922208152212,
    2.480141094633707,
    -1.392156691525032,
    1.284644859983834
   ],
   [
    -1.0849963573969008,
    -1.3464025030099478,
    -1.5806693812349994,
    0.09313312001258478,
    0.5882369609009444,
    0.12299600253327525
   ],
   [
    -0.12813652903419187,
    -0.11551941969968008,
    -0.0027257909744386076,
    -0.18599044244708512,
    0.10987367870891698,
    -0.035353787347586166
   ],
   [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ]
  ],
  "dW_partial": [
   [
    2.112745638322741,
    4.358200871476191,
    1.7279641232347078,
    0.14185824172794076,
    -0.21914809289842704
   ],
   [
    1.2996974203659213,
    -2.15152130457065,
    1.130451848843385,
    -0.35058515516938815,
    1.0392523067170458
   ],
   [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ],
   [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ],
   [
    -0.18414107808213095,
    0.12139749316708905,
    -0.07383878123140582,
    0.12317263869373725,
    -0.06395689181074668
   ],
   [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
   ]
  ],
  "delta_bias": [
   -1.8789206743515927,
   -0.04423040979824179,
   1.284644859983834,
   0.12299600253327525,
   -0.035353787347586166,
   0.0
  ],
  "rank_clamp": [
   {
    "subspace_rank": 1,
    "seg_lens": [
     2,
     2,
     1
    ],
    "actual": [
     1,
     1,
     1
    ]
   },
   {
    "subspace_rank": 2,
    "seg_lens": [
     2,
     2,
     1
    ],
    "actual": [
     2,
     2,
     1
    ]
   },
   {
    "subspace_rank": 5,
    "seg_lens": [
     2,
     2,
     1
    ],
    "actual": [
     2,
     2,
     1
    ]
   },
   {
    "subspace_rank": 3,
    "seg_lens": [
     0
    ],
    "actual": [
     1
    ]
   }
  ]
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
    seg = {int(k): v for k, v in f["seg"].items()}
    newB = np.array(f["newB"])
    A = [np.array(a) for a in f["A"]]
    B = [np.array(b) for b in f["B"]]
    x = np.array(f["x"])
    return seg, newB, A, B, x


# ---------- mode="build_scatter_index" / "rank_clamp" ----------


def check_scatter_rank(solve):
    f = EXPECTED["F1"]
    seg = {int(k): v for k, v in f["seg"].items()}
    # 全训: 按训练组顺序拼接，覆盖全部 r 个成分
    got = solve(mode="build_scatter_index", seg=seg, train_idx=f["train_idx"])
    assert got == f["scatter_index"], got
    # 部分训练: 跳过组 1，中段成分行零散射
    got_p = solve(mode="build_scatter_index", seg=seg, train_idx=[0, 2])
    assert got_p == f["scatter_index_partial"], got_p
    # 顺序敏感: 反序训练组得到不同索引
    assert got_p != solve(mode="build_scatter_index", seg=seg, train_idx=[2, 0])
    # rank_clamp 账: actual = max(1, min(seg_len, subspace_rank))
    for case in f["rank_clamp"]:
        got_r = solve(mode="rank_clamp", seg_lens=case["seg_lens"],
                      subspace_rank=case["subspace_rank"])
        assert got_r == case["actual"], (case, got_r)
    # 空 seg 段护栏: actual 至少 1
    g = solve(mode="rank_clamp", seg_lens=[0], subspace_rank=3)
    assert g == [1], g


# ---------- mode="forward_delta" ----------


def check_forward_delta(solve):
    f = EXPECTED["F1"]
    seg, newB, A, B, x = _fix()
    sc = f["scatter_index"]
    got = solve(mode="forward_delta", x=x, newB=newB, A_list=A, B_list=B,
                scatter_index=sc, out_dim=f["out_dim"])
    _assert_close_3d(got, f["forward"])
    # 互证: forward == x @ dW.T（merge 与 forward 同源）
    dW = np.array(f["dW"])
    _assert_close_3d(got, (x @ dW.T).tolist())
    # 未训练行（成分 5 无 delta）保持零
    assert np.allclose(np.asarray(got)[..., f["out_dim"] - 1], 0.0)
    # 部分训练: 组 1 对应行 [2, 3] 恒零
    got_p = solve(mode="forward_delta", x=x, newB=newB, A_list=[A[0], A[2]],
                  B_list=[B[0], B[2]], scatter_index=f["scatter_index_partial"],
                  out_dim=f["out_dim"])
    _assert_close_3d(got_p, f["forward_partial"])
    assert np.allclose(np.asarray(got_p)[..., 2] + np.asarray(got_p)[..., 3], 0.0)


# ---------- mode="forward_delta" with bias ----------


def check_forward_bias(solve):
    f = EXPECTED["F1"]
    seg, newB, A, B, x = _fix()
    sc = f["scatter_index"]
    newB_b = np.array(f["newB_b"])
    got = solve(mode="forward_delta", x=x, newB=newB_b, A_list=A, B_list=B,
                scatter_index=sc, out_dim=f["out_dim"], with_bias=True)
    _assert_close_3d(got, f["forward_b"])
    # 输出维 = out_dim（bias 折进输入侧全 1 列，不是输出 +1）
    assert np.asarray(got).shape[-1] == f["out_dim"]
    # 互证: x @ W_delta.T + b_delta == forward（split 后两件套）
    dW_full = np.array(f["dW_full"])
    W, b = solve(mode="split_delta", dW_full=dW_full, with_bias=True)
    assert np.allclose(np.asarray(W), dW_full[:, :-1])
    _assert_close_3d(got, (x @ np.asarray(W).T + np.asarray(b)).tolist())
    # 维度护栏: no-bias 输入 x (in) 配含 bias 列的 newB (r, in+1) -> ValueError
    try:
        solve(mode="forward_delta", x=x, newB=newB_b, A_list=A, B_list=B,
              scatter_index=sc, out_dim=f["out_dim"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------- mode="get_delta_full" ----------


def check_get_delta_full(solve):
    f = EXPECTED["F1"]
    seg, newB, A, B, x = _fix()
    sc = f["scatter_index"]
    got = solve(mode="get_delta_full", newB=newB, A_list=A, B_list=B,
                scatter_index=sc, out_dim=f["out_dim"])
    _assert_close_2d(got, f["dW"])
    # 增广形态 (out, in+1) 含 bias 列
    got_b = solve(mode="get_delta_full", newB=np.array(f["newB_b"]), A_list=A,
                  B_list=B, scatter_index=sc, out_dim=f["out_dim"])
    _assert_close_2d(got_b, f["dW_full"])
    assert np.asarray(got_b).shape == (f["out_dim"], f["in_dim"] + 1)
    # 未训练行零: 成分 5 无 delta
    assert np.allclose(np.asarray(got)[f["out_dim"] - 1], 0.0)
    # 部分训练: 组 1 行 [2, 3] 零
    got_p = solve(mode="get_delta_full", newB=newB, A_list=[A[0], A[2]],
                  B_list=[B[0], B[2]], scatter_index=f["scatter_index_partial"],
                  out_dim=f["out_dim"])
    _assert_close_2d(got_p, f["dW_partial"])
    assert np.allclose(np.asarray(got_p)[2] + np.asarray(got_p)[3], 0.0)


# ---------- mode="split_delta" ----------


def check_split_delta(solve):
    f = EXPECTED["F1"]
    dW_full = np.array(f["dW_full"])
    W, b = solve(mode="split_delta", dW_full=dW_full, with_bias=True)
    _assert_close_2d(W, (np.array(f["dW_full"])[:, :-1]).tolist())
    _assert_close_list([float(v) for v in b], f["delta_bias"])
    assert np.asarray(W).shape == (f["out_dim"], f["in_dim"])
    assert np.asarray(b).shape == (f["out_dim"],)
    # no-bias 分支: weight 即全量，bias 全零
    dW_nb = np.array(f["dW"])
    W2, b2 = solve(mode="split_delta", dW_full=dW_nb, with_bias=False)
    _assert_close_2d(W2, f["dW"])
    assert np.allclose(np.asarray(b2), 0.0)


# ---------- solve dispatch ----------


def check_dispatch(solve):
    try:
        solve(mode="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


CHECKS = [
    check_scatter_rank,
    check_forward_delta,
    check_forward_bias,
    check_get_delta_full,
    check_split_delta,
    check_dispatch,
]
