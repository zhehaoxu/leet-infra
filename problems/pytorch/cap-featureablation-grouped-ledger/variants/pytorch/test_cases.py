#!/usr/bin/env python3
"""856 judge cases: Captum FeatureAblation grouped-ablation ledger.

Every check takes `solve` and asserts on explicit, hand-verified values that
were produced by the verbatim official implementation (see notes.md).
No torch, no imports from the solution module, no global RNG.
"""


def _near(got, exp, tol=1e-8):
    if isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), (got, exp)
        for a, b in zip(got, exp):
            _near(a, b, tol)
    elif isinstance(exp, float):
        assert abs(got - exp) <= tol, (got, exp)
    else:
        assert got == exp, (got, exp)


def _err(fn, key):
    try:
        fn()
    except ValueError as e:
        assert key in str(e), "expected %r in %r" % (key, str(e))
        return
    raise AssertionError("expected ValueError containing %r" % key)


# ---------------------------------------------------------------------------
def check_mask_range(solve):
    """Default/offset masks, max+1 vs max-min+1, clamped ppe, progress total."""
    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]])
    _near(r["mask"], [[0, 1, 2]])
    _near(r["mask_shape"], [1, 3])
    assert r["numel_per_example"] == 3
    assert r["min_feature"] == 0 and r["num_features"] == 3
    assert r["feature_count"] == 3 and r["loop_start"] == 0
    assert r["num_steps"] == 3 and r["ppe_clamped"] == 1
    _near(r["chunks"], [[0, 1], [1, 2], [2, 3]])
    assert r["num_chunks"] == 3 and r["progress_total"] == 4

    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]],
              feature_mask=[[5, 5, 7]], perturbations_per_eval=2)
    assert r["min_feature"] == 5, r["min_feature"]
    assert r["num_features"] == 8, r["num_features"]
    assert r["feature_count"] == 3, r["feature_count"]
    assert r["loop_start"] == 5 and r["num_steps"] == 3
    assert r["ppe_clamped"] == 2
    _near(r["chunks"], [[5, 7], [7, 8]])
    assert r["num_chunks"] == 2
    assert r["progress_total"] == 3, r["progress_total"]

    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]],
              feature_mask=[[0, 0, 3]], perturbations_per_eval=2)
    assert r["min_feature"] == 0 and r["num_features"] == 4
    assert r["feature_count"] == 4
    assert r["loop_start"] == 0 and r["num_steps"] == 4
    _near(r["chunks"], [[0, 2], [2, 4]])

    r = solve("mask_range", input=[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
              perturbations_per_eval=2)
    _near(r["mask"], [[[0, 1], [2, 3]]])
    _near(r["mask_shape"], [1, 2, 2])
    assert r["numel_per_example"] == 4 and r["num_features"] == 4
    _near(r["chunks"], [[0, 2], [2, 4]])
    assert r["progress_total"] == 3

    r = solve("mask_range", input=[[1., 2.], [3., 4.]], offset=3,
              perturbations_per_eval=1)
    _near(r["mask"], [[3, 4]])
    assert r["min_feature"] == 3 and r["num_features"] == 5
    _near(r["chunks"], [[3, 4], [4, 5]])

    # official isinstance(True, int) quirk: bool ppe is accepted as 1
    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]],
              perturbations_per_eval=True)
    assert r["ppe_clamped"] == 1 and r["num_chunks"] == 3

    # a trailing mask dimension of 1 is broadcastable (official _is_mask_valid)
    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]],
              feature_mask=[[0]], perturbations_per_eval=1)
    _near(r["mask"], [[0]])
    _near(r["mask_shape"], [1, 1])
    assert r["numel_per_example"] == 3
    assert r["num_features"] == 1 and r["num_steps"] == 1
    _near(r["chunks"], [[0, 1]])

    # ppe larger than num_features: the clamp uses max+1, not max-min+1
    r = solve("mask_range", input=[[1., 2., 3.], [4., 5., 6.]],
              feature_mask=[[5, 5, 7]], perturbations_per_eval=10)
    assert r["ppe_clamped"] == 8, r["ppe_clamped"]
    assert r["num_features"] == 8 and r["feature_count"] == 3
    _near(r["chunks"], [[5, 8]])


# ---------------------------------------------------------------------------
def check_mask_range_domain(solve):
    """Input contracts and the unknown-mode dispatch."""
    inp = [[1., 2., 3.], [4., 5., 6.]]
    for bad in (0, -1, 1.5, "2", None):
        _err(lambda b=bad: solve("mask_range", input=inp,
                                 perturbations_per_eval=b),
             "perturbations_per_eval must be an integer and at least 1")
    _err(lambda: solve("mask_range", input=[1., 2., 3.]),
         "input must have a leading example dimension")
    _err(lambda: solve("mask_range", input=inp, offset=-1),
         "offset must be non-negative")
    _err(lambda: solve("mask_range", input=inp,
                       feature_mask=[[[0, 1, 2]]]),
         "feature mask must have the same rank as the input")
    _err(lambda: solve("mask_range", input=inp,
                       feature_mask=[[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
         "feature mask first dim must be 1 or num_examples")
    _err(lambda: solve("mask_range", input=inp, feature_mask=[[0, 1]]),
         "feature mask shape is not broadcastable to input")
    _err(lambda: solve("mask_range", input=inp, feature_mask=[[0, 1.5, 2]]),
         "feature mask values must be non-negative integers")
    _err(lambda: solve("mask_range", input=inp, feature_mask=[[0, -1, 2]]),
         "feature mask values must be non-negative integers")
    _err(lambda: solve("mask_range", input=inp, feature_mask=[[0, True, 2]]),
         "feature mask values must be non-negative integers")
    _err(lambda: solve("nope", input=inp), "unknown mode")


# ---------------------------------------------------------------------------
def check_ablate(solve):
    """_construct_ablated_input: stacked masks, baseline blending, holes."""
    e = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    r = solve("ablate", expanded_input=e, input_mask=[[0, 1]],
              baseline=0.5, start_feature=0, end_feature=2)
    _near(r["mask"], [[[1, 0]], [[0, 1]]])
    _near(r["mask_shape"], [2, 1, 2])
    _near(r["ablated"], [[[0.5, 2.], [0.5, 4.]],
                         [[5., 0.5], [7., 0.5]]])
    _near(r["ablated_shape"], [2, 2, 2])
    assert r["num_ablated_features"] == 2
    _near(r["ones_per_feature"], [1, 1])
    _near(r["baseline_shape"], [])

    r = solve("ablate", expanded_input=e, input_mask=[[0, 1]],
              baseline=[[9., 9.], [9., 9.]], start_feature=0, end_feature=2)
    _near(r["ablated"], [[[9., 2.], [9., 4.]],
                         [[5., 9.], [7., 9.]]])
    _near(r["baseline_shape"], [2, 2])

    r = solve("ablate", expanded_input=[[[1., 2., 3.]], [[4., 5., 6.]]],
              input_mask=[[0, 1, 1]], baseline=-1.0,
              start_feature=3, end_feature=5)
    _near(r["mask"], [[[0, 0, 0]], [[0, 0, 0]]])
    _near(r["ablated"], [[[1., 2., 3.]], [[4., 5., 6.]]])
    _near(r["ones_per_feature"], [0, 0])

    r = solve("ablate",
              expanded_input=[[[1., 2.], [3., 4.]],
                              [[5., 6.], [7., 8.]],
                              [[9., 10.], [11., 12.]]],
              input_mask=[[0, 1]], baseline=2.0,
              start_feature=0, end_feature=3)
    _near(r["mask"], [[[1, 0]], [[0, 1]], [[0, 0]]])
    _near(r["ablated"], [[[2., 2.], [2., 4.]],
                         [[5., 2.], [7., 2.]],
                         [[9., 10.], [11., 12.]]])
    _near(r["ones_per_feature"], [1, 1, 0])

    # default baseline is the scalar 0
    r = solve("ablate", expanded_input=e, input_mask=[[0, 1]],
              start_feature=0, end_feature=2)
    _near(r["ablated"], [[[0., 2.], [0., 4.]], [[5., 0.], [7., 0.]]])
    _near(r["baseline_shape"], [])

    # a row-dependent tensor baseline exercises the leading-dim broadcast
    r = solve("ablate", expanded_input=e, input_mask=[[0, 1]],
              baseline=[[9., 9.], [1., 1.]],
              start_feature=0, end_feature=2)
    _near(r["ablated"], [[[9., 2.], [1., 4.]], [[5., 9.], [7., 1.]]])


# ---------------------------------------------------------------------------
def check_ablate_domain(solve):
    """_construct_ablated_input contracts."""
    e = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
    _err(lambda: solve("ablate", expanded_input=[1., 2.], input_mask=[[0, 1]],
                       start_feature=0, end_feature=2),
         "expanded_input must have shape (n_features, n_examples, ...)")
    _err(lambda: solve("ablate", expanded_input=e, input_mask=[[0., 1.]],
                       start_feature=0, end_feature=2),
         "input_mask values must be integers")
    _err(lambda: solve("ablate", expanded_input=e, input_mask=[[0, 1]],
                       start_feature=0.5, end_feature=2),
         "start_feature and end_feature must be integers")
    _err(lambda: solve("ablate", expanded_input=e, input_mask=[[0, 1]],
                       start_feature=2, end_feature=2),
         "start_feature must be less than end_feature")
    _err(lambda: solve("ablate", expanded_input=e, input_mask=[[0, 1]],
                       start_feature=-1, end_feature=2),
         "start_feature must be non-negative")
    _err(lambda: solve("ablate", expanded_input=e,
                       input_mask=[[0, 1, 2]],
                       start_feature=0, end_feature=2),
         "input_mask is not broadcastable to expanded_input")
    _err(lambda: solve("ablate", expanded_input=e,
                       input_mask=[[[0, 1]]],
                       start_feature=0, end_feature=2),
         "input_mask broadcast changes the expanded_input shape")


# ---------------------------------------------------------------------------
def check_schedule(solve):
    """_ith_input_ablation_generator: repetition, tail chunks, expansions."""
    r = solve("schedule",
              inputs=[[[1., 2., 3.], [4., 5., 6.]],
                      [[7., 8.], [9., 10.]]],
              feature_mask=[[[0, 0, 1]], [[0, 1]]],
              baselines=[0., 0.], index=0, perturbations_per_eval=1)
    assert r["ppe_effective"] == 1 and r["num_examples"] == 2
    assert r["num_features"] == 2 and r["loop_start"] == 0
    assert r["num_chunks"] == 2 and r["repeated_batch_size"] == 2
    _near(r["chunks"], [[0, 1], [1, 2]])
    assert r["progress_total"] == 5
    ys = r["yields"]
    _near(ys[0]["current_inputs"][0], [[0., 0., 3.], [0., 0., 6.]])
    _near(ys[0]["current_inputs"][1], [[7., 8.], [9., 10.]])
    _near(ys[0]["current_mask"], [[[1, 1, 0]]])
    assert ys[0]["batch_size"] == 2 and ys[0]["start"] == 0
    _near(ys[1]["current_inputs"][0], [[1., 2., 0.], [4., 5., 0.]])
    _near(ys[1]["current_mask"], [[[0, 0, 1]]])

    r = solve("schedule",
              inputs=[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
              feature_mask=[[[0, 1]], [[0, 1]]],
              baselines=[0., 0.], index=1, perturbations_per_eval=1)
    _near(r["yields"][0]["current_inputs"][0], [[1., 2.], [3., 4.]])
    _near(r["yields"][0]["current_inputs"][1], [[0., 6.], [0., 8.]])
    assert r["extra_args_selected"] == {}

    r = solve("schedule",
              inputs=[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]],
              perturbations_per_eval=3, target=[1, 0],
              additional_forward_args=[2])
    assert r["ppe_effective"] == 3 and r["num_features"] == 5
    assert r["repeated_batch_size"] == 6
    _near(r["chunks"], [[0, 3], [3, 5]])
    assert r["num_chunks"] == 2 and r["progress_total"] == 3
    _near(r["yields"][0]["current_target"], [1, 0, 1, 0, 1, 0])
    _near(r["yields"][0]["current_additional_args"], [2])
    _near(r["yields"][0]["current_inputs"][0][:3],
          [[0., 2., 3., 4., 5.], [0., 7., 8., 9., 10.],
           [1., 0., 3., 4., 5.]])
    _near(r["yields"][0]["current_mask"],
          [[[1, 0, 0, 0, 0]], [[0, 1, 0, 0, 0]], [[0, 0, 1, 0, 0]]])
    _near(r["yields"][1]["current_target"], [1, 0, 1, 0])
    _near(r["yields"][1]["current_mask"],
          [[[0, 0, 0, 1, 0]], [[0, 0, 0, 0, 1]]])
    assert r["yields"][1]["batch_size"] == 4
    assert r["yields"][1]["k"] == 2
    _near(r["yields"][1]["current_inputs"][0],
          [[1., 2., 3., 0., 5.], [6., 7., 8., 0., 10.],
           [1., 2., 3., 4., 0.], [6., 7., 8., 9., 0.]])

    r = solve("schedule",
              inputs=[[[1., 2., 3.], [4., 5., 6.]], [[7., 8.], [9., 10.]]],
              feature_mask=[[[0, 0, 1]], [[0, 1]]],
              baselines=[0., 0.], index=0, perturbations_per_eval=2,
              additional_forward_args=[3, [[1.], [2.]]], target=[1, 0])
    assert r["ppe_effective"] == 2
    _near(r["yields"][0]["current_inputs"][1],
          [[7., 8.], [9., 10.], [7., 8.], [9., 10.]])
    _near(r["yields"][0]["current_additional_args"],
          [3, [[1.], [2.], [1.], [2.]]])
    _near(r["yields"][0]["current_target"], [1, 0, 1, 0])

    # min>0 + oversized ppe: ppe_eff uses max+1, chunk count uses max-min+1
    r = solve("schedule",
              inputs=[[[1., 2., 3., 4.], [5., 6., 7., 8.]]],
              feature_mask=[[[10, 10, 12, 12]]], baselines=[0.],
              index=0, perturbations_per_eval=5, target=[1, 0])
    assert r["ppe_effective"] == 5 and r["num_features"] == 13
    assert r["feature_count"] == 3 and r["loop_start"] == 10
    assert r["num_chunks"] == 1
    _near(r["chunks"], [[10, 13]])
    _near(r["yields"][0]["current_mask"],
          [[[1, 1, 0, 0]], [[0, 0, 0, 0]], [[0, 0, 1, 1]]])
    _near(r["yields"][0]["current_target"], [1, 0, 1, 0, 1, 0])

    r = solve("schedule",
              inputs=[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
              feature_mask=[[[0, 1]], [[0, 1]]],
              baselines=[0., 0.], index=1, perturbations_per_eval=1,
              extra_args={"stride": (11, 22), "scale": 4})
    assert r["extra_args_selected"] == {"stride": 22, "scale": 4}

    r = solve("schedule",
              inputs=[[[1., 2.], [3., 4.]]],
              baselines=[[[2., 3.], [2., 3.]]],
              perturbations_per_eval=1)
    _near(r["baseline_reshape"], [1, 2, 2])
    _near(r["yields"][0]["current_inputs"][0],
          [[2., 2.], [2., 4.]])

    # ppe above num_features is clamped; the metadata exposes the clamp
    r = solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
              perturbations_per_eval=5)
    assert r["ppe_effective"] == 2, r["ppe_effective"]
    assert r["repeated_batch_size"] == 4
    assert r["num_features"] == 2
    assert r["yields"][0]["k"] == 2
    _near(r["yields"][0]["current_inputs"][0],
          [[0., 2.], [0., 4.], [1., 0.], [3., 0.]])

    # a tail chunk re-expands tensor additional args by K, not by ppe_eff
    r = solve("schedule",
              inputs=[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]],
              perturbations_per_eval=3, target=[1, 0],
              additional_forward_args=[3, [[1.], [2.]]])
    _near(r["yields"][0]["current_additional_args"],
          [3, [[1.], [2.], [1.], [2.], [1.], [2.]]])
    _near(r["yields"][1]["current_additional_args"],
          [3, [[1.], [2.], [1.], [2.]]])

    # a tail chunk slices EVERY repeated input, not just the ablated one
    r = solve("schedule",
              inputs=[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]],
                      [[7., 8.], [9., 10.]]],
              baselines=[0., 0.], index=0, perturbations_per_eval=3)
    assert r["num_chunks"] == 2
    _near(r["yields"][0]["current_inputs"][1],
          [[7., 8.], [9., 10.], [7., 8.], [9., 10.],
           [7., 8.], [9., 10.]])
    _near(r["yields"][1]["current_inputs"][1],
          [[7., 8.], [9., 10.], [7., 8.], [9., 10.]])

    # min>0: progress uses max-min+1, never max+1
    r = solve("schedule",
              inputs=[[[1., 2., 3., 4.], [5., 6., 7., 8.]]],
              feature_mask=[[[10, 10, 12, 12]]], baselines=[0.],
              index=0, perturbations_per_eval=5, target=[1, 0])
    assert r["progress_total"] == 2, r["progress_total"]


# ---------------------------------------------------------------------------
def check_schedule_domain(solve):
    """Generator-level input contracts."""
    _err(lambda: solve("schedule", inputs=[], perturbations_per_eval=1),
         "inputs must be a non-empty list of input arrays")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       feature_mask=[[[0, 1]], [[0, 1]]]),
         "feature mask count must match inputs")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       baselines=[0., 0.]),
         "baselines count must match inputs")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       baselines=["x"]),
         "baseline must be a number or a broadcastable array")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       additional_forward_args=5),
         "additional_forward_args must be a list")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]], index=3),
         "index out of range for inputs")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       feature_mask=[[[0, 1.5]]]),
         "feature mask values must be non-negative integers")
    _err(lambda: solve("schedule", inputs=[[[1., 2.], [3., 4.]]],
                       baselines=[[[1., 2.], [1., 2.], [1., 2.]]]),
         "baseline first dim must be 1 or num_examples")


# ---------------------------------------------------------------------------
def check_eval_out(solve):
    """_process_initial_eval / _process_ablated_out / _generate_result."""
    r = solve("eval_out", initial_eval=[14., 32.], input_shapes=[[2, 3]],
              modified_evals=[[9., 18.], [5., 14.]],
              masks=[[[[1, 1, 0]]], [[[0, 0, 1]]]],
              batch_sizes=[2, 2], perturbations_per_eval=1,
              num_examples=2)
    _near(r["attributions"], [[[5., 5., 9.], [14., 14., 18.]]])
    assert r["n_outputs"] == 2 and r["attrib_type"] == "float64"
    _near(r["flattened_initial_eval"], [[14., 32.]])
    _near(r["total_attrib_shapes"], [[2, 3]])
    _near(r["initial_shape"], [2])
    _near(r["eval_diffs"], [[[[5.], [14.]]], [[[9.], [18.]]]])
    assert r["weights"] == []

    r = solve("eval_out", initial_eval=[3., 7.], input_shapes=[[2, 2]],
              modified_evals=[[1., 2.]],
              masks=[[[[1, 1]], [[1, 0]]]],
              batch_sizes=[2], perturbations_per_eval=1,
              num_examples=2, use_weights=True)
    _near(r["attributions"], [[[2., 2.], [5., 5.]]])
    _near(r["weights"], [[[2., 1.], [2., 1.]]])

    r = solve("eval_out", initial_eval=10, input_shapes=[[2, 2]],
              modified_evals=[6, 4],
              masks=[[[[1, 0]]], [[[0, 1]]]],
              batch_sizes=[2, 2], perturbations_per_eval=1,
              num_examples=2)
    _near(r["attributions"], [[[4, 6]]])
    assert r["attrib_type"] == "int64" and r["n_outputs"] == 1
    _near(r["flattened_initial_eval"], [[10]])

    r = solve("eval_out", initial_eval=10.5, input_shapes=[[2, 2]],
              modified_evals=[6.5],
              masks=[[[[1, 0]]]],
              batch_sizes=[2], perturbations_per_eval=1,
              num_examples=2)
    assert r["attrib_type"] == "float64"
    _near(r["attributions"], [[[4.0, 0.0]]])

    # no records: accumulator shapes and initial shape stay untouched
    r = solve("eval_out", initial_eval=[14., 32.], input_shapes=[[2, 3]],
              modified_evals=[], masks=[], batch_sizes=[],
              perturbations_per_eval=1, num_examples=2)
    _near(r["total_attrib_shapes"], [[2, 3]])
    _near(r["initial_shape"], [2])
    _near(r["attributions"], [[[0, 0, 0], [0, 0, 0]]])

    # one ppe>1 record with two perturbation rows
    r = solve("eval_out", initial_eval=[14., 32.], input_shapes=[[2, 3]],
              modified_evals=[[9., 18., 5., 14.]],
              masks=[[[[1, 1, 0]], [[0, 0, 1]]]],
              batch_sizes=[4], perturbations_per_eval=2, num_examples=2)
    _near(r["attributions"], [[[5., 5., 9.], [14., 14., 18.]]])
    _near(r["eval_diffs"], [[[[5.], [14.]], [[9.], [18.]]]])


# ---------------------------------------------------------------------------
def check_eval_out_domain(solve):
    """Accumulator contracts and the ppe>1 output-shape assertion."""
    _err(lambda: solve("eval_out", initial_eval=[1., 2.], input_shapes=[],
                       modified_evals=[], masks=[], batch_sizes=[]),
         "input_shapes must be a non-empty list of shapes")
    _err(lambda: solve("eval_out", initial_eval=[1., 2.], input_shapes=[[2, 2]],
                       index=4, modified_evals=[], masks=[], batch_sizes=[]),
         "index out of range for input_shapes")
    _err(lambda: solve("eval_out", initial_eval=[1., 2.], input_shapes=[[2, 2]],
                       modified_evals=[[1., 2.]], masks=[], batch_sizes=[]),
         "modified_evals, masks and batch_sizes lengths must agree")
    _err(lambda: solve("eval_out", initial_eval=True, input_shapes=[[2, 2]],
                       modified_evals=[], masks=[], batch_sizes=[]),
         "initial_eval must be a number or an array")
    _err(lambda: solve("eval_out", initial_eval=[1., 2.], input_shapes=[[2, 2]],
                       modified_evals=[[1., 2.]], masks=[[[[1, 0]]]],
                       batch_sizes=[4], perturbations_per_eval=2,
                       num_examples=2),
         "output should be a tensor whose 1st dim grow")
    _err(lambda: solve("eval_out", initial_eval=[1., 2.], input_shapes=[[2, 2]],
                       modified_evals=[[1., 2., 3.]], masks=[[[[1, 0]]]],
                       batch_sizes=[2], perturbations_per_eval=1,
                       num_examples=2),
         "modified eval cannot reshape to (-1, n_outputs)")


# ---------------------------------------------------------------------------
def check_attribution(solve):
    """End-to-end FeatureAblation.attribute ledger."""
    r = solve("attribution", inputs=[[[1., 2., 3.], [4., 5., 6.]]],
              feature_mask=[[[0, 0, 1]]], baselines=[0.],
              forward_weights=[1., 2., 3.], perturbations_per_eval=1)
    _near(r["attributions"], [[5., 5., 9.], [14., 14., 18.]])
    assert r["n_outputs"] == 2 and r["attrib_type"] == "float64"
    _near(r["flattened_initial_eval"], [[14., 32.]])
    assert r["num_forwards"] == 3
    _near(r["forward_shapes"], [[2], [2], [2]])
    _near(r["chunks"], [{"input_index": 0, "start": 0, "end": 1, "batch_size": 2},
                        {"input_index": 0, "start": 1, "end": 2, "batch_size": 2}])
    assert r["progress_total"] is None

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[1., 2.], forward_kind="quadratic")
    _near(r["attributions"], [[1., 8.], [9., 32.]])
    _near(r["flattened_initial_eval"], [[9., 41.]])

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[1., 1.], baselines=[2.])
    _near(r["attributions"], [[-1., 0.], [1., 2.]])

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[[[1., 2.], [2., 1.]]],
              forward_kind="multi", target=1, n_classes=2)
    _near(r["attributions"], [[2., 2.], [6., 4.]])
    _near(r["flattened_initial_eval"], [[4., 10.]])

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[[[1., 2.], [2., 1.]]],
              forward_kind="multi", target=[1, 0], n_classes=2,
              perturbations_per_eval=2)
    _near(r["attributions"], [[2., 2.], [3., 8.]])
    _near(r["flattened_initial_eval"], [[4., 11.]])
    assert r["num_forwards"] == 2
    _near(r["forward_shapes"], [[2], [4]])

    for kind, out in (("aggregate", [[4., 6.]]),
                      ("python_int", [[4, 6]]),
                      ("python_float", [[4., 6.]])):
        r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                  forward_weights=[1., 1.], forward_kind=kind)
        _near(r["attributions"], out)
        assert r["n_outputs"] == 1
        _near(r["initial_shape"], [])
        _near(r["forward_shapes"], [[], [], []])
        if kind == "python_int":
            assert isinstance(r["attributions"][0][0], int)
            assert isinstance(r["flattened_initial_eval"][0][0], int)
        if kind == "python_float":
            assert isinstance(r["attributions"][0][0], float)
            assert isinstance(r["flattened_initial_eval"][0][0], float)

    r = solve("attribution",
              inputs=[[[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]]],
              forward_weights=[1., 1., 1., 1., 1.],
              perturbations_per_eval=3)
    assert r["num_forwards"] == 3
    _near(r["forward_shapes"], [[2], [6], [4]])
    _near(r["attributions"],
          [[1., 2., 3., 4., 5.], [6., 7., 8., 9., 10.]])

    r = solve("attribution", inputs=[[[1., 2., 3.], [4., 5., 6.]]],
              feature_mask=[[[0, 0, 1]]],
              forward_weights=[1., 2., 3.],
              perturbations_per_eval=2, show_progress=True)
    _near(r["attributions"], [[5., 5., 9.], [14., 14., 18.]])
    assert r["progress_total"] == 2
    assert r["num_forwards"] == 2

    r = solve("attribution", inputs=[[[1., 2., 3.], [4., 5., 6.]]],
              feature_mask=[[[10, 10, 12]]],
              forward_weights=[1., 2., 3.],
              perturbations_per_eval=4)
    _near(r["attributions"], [[5., 5., 9.], [14., 14., 18.]])
    _near(r["chunks"], [{"input_index": 0, "start": 10, "end": 13,
                         "batch_size": 6}])

    r = solve("attribution",
              inputs=[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]],
              forward_weights=[[1., 2.], [1., 1.]], tuple_inputs=True)
    _near(r["attributions"],
          [[[1., 4.], [3., 8.]], [[5., 6.], [7., 8.]]])
    assert r["num_forwards"] == 5
    _near(r["chunks"],
          [{"input_index": 0, "start": 0, "end": 1, "batch_size": 2},
           {"input_index": 0, "start": 1, "end": 2, "batch_size": 2},
           {"input_index": 1, "start": 2, "end": 3, "batch_size": 2},
           {"input_index": 1, "start": 3, "end": 4, "batch_size": 2}])

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              feature_mask=[[[0, 1]]], baselines=[0.],
              forward_weights=[1., 1.], use_weights=True)
    _near(r["attributions"], [[1., 2.], [3., 4.]])
    _near(r["weights"], [[[1., 1.], [1., 1.]]])

    r = solve("attribution", inputs=[[[1., 2., 3.], [4., 5., 6.]]],
              forward_weights=[1., 2., 3.],
              additional_forward_args=[0.5])
    _near(r["flattened_initial_eval"], [[14.5, 32.5]])
    assert r["num_forwards"] == 4

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              feature_mask=[[[0, 1]]],
              baselines=[[[2., 3.], [2., 3.]]],
              forward_weights=[1., 1.])
    _near(r["attributions"], [[-1., -1.], [1., 1.]])

    # scalar forward with an additive bias: only the flattened eval changes
    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[1., 1.], forward_kind="aggregate",
              additional_forward_args=[0.5])
    _near(r["flattened_initial_eval"], [[10.5]])
    _near(r["attributions"], [[4., 6.]])

    r = solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
              forward_weights=[[[1., 2.], [2., 1.]]],
              forward_kind="multi", target=1,
              additional_forward_args=[0.5])
    _near(r["flattened_initial_eval"], [[4.5, 10.5]])
    _near(r["initial_shape"], [2])

    # min>0 mask with show_progress: ceil((max-min+1)/ppe)+1
    r = solve("attribution", inputs=[[[1., 2., 3.], [4., 5., 6.]]],
              feature_mask=[[[10, 10, 12]]],
              forward_weights=[1., 2., 3.],
              perturbations_per_eval=4, show_progress=True)
    assert r["progress_total"] == 2, r["progress_total"]
    _near(r["attributions"], [[5., 5., 9.], [14., 14., 18.]])

    # official skips empty input tensors entirely
    r = solve("attribution",
              inputs=[[[1., 2.], [3., 4.]], [[], []]],
              forward_weights=[[1., 2.], []], tuple_inputs=True)
    _near(r["attributions"], [[[1., 4.], [3., 8.]], [[], []]])
    assert r["num_forwards"] == 3


# ---------------------------------------------------------------------------
def check_attribution_domain(solve):
    """End-to-end input contracts and dispatch errors."""
    _err(lambda: solve("attribution", inputs=[],
                       forward_weights=[1.]),
         "inputs must be a non-empty list of input arrays")
    _err(lambda: solve("attribution",
                       inputs=[[[1., 2.]], [[3., 4.]]],
                       forward_weights=[1., 1.]),
         "single-input attribution needs exactly one input array")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[1., 1.], forward_kind="nope"),
         "unknown forward_kind")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[1., 1.],
                       additional_forward_args=5),
         "additional_forward_args must be a list")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[1., 1.],
                       additional_forward_args=[[1.], [2.]]),
         "additional_forward_args must be scalar numbers in attribution mode")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[[[1., 2.], [2., 1.]]],
                       forward_kind="multi", target=[1, 0, 1]),
         "Target list length does not match output!")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[[[1., 2.], [2., 1.]]],
                       forward_kind="multi", target=None),
         "multi forward needs an integer or list target")
    _err(lambda: solve("attribution", inputs=[[[1., 2.], [3., 4.]]],
                       forward_weights=[1., 1.], forward_kind="aggregate",
                       perturbations_per_eval=2),
         "output should be a tensor whose 1st dim grow")
