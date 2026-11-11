#!/usr/bin/env python3
"""865 judge cases: torchaudio SpecAugment mask ledger.

Constants below are exact snapshots produced by fixtures865.py from the
verbatim official implementation. Helpers use internal assert, never a
returned bool. No torch, no import of the solution module.
"""

# ---- official numerical fixtures ----
DEFAULT_PARAM = 3

DEFAULT_RAND_CALLS = [{'dtype': 'float64', 'shape': [2], 'values': [0.31, 0.48]},
 {'dtype': 'float64', 'shape': [2], 'values': [0.13, 0.42]}]

DEFAULT_INTERVAL = [{'end': 0, 'min_value': 0.39910000000000007, 'start': 0, 'value': 0.9299999999999999, 'width': 0},
 {'end': 2, 'min_value': 1.0752, 'start': 1, 'value': 1.44, 'width': 1}]

DEFAULT_MASK = [[0, 0, 0, 0], [0, 1, 0, 0]]

DEFAULT_OUTPUT = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

DEFAULT_IDENTITY = {'err': None, 'rand_calls': 2, 'same_object': False}

DEFAULT_ERRORS = {'axis_negative': 'ValueError: Only Frequency and Time masking are supported (axis 1 and axis 2 '
                  'supported; -1 given).',
 'axis_oob': 'ValueError: Only Frequency and Time masking are supported (axis 1 and axis 2 supported; '
             '0 given).',
 'dim_iid': 'ValueError: Spectrogram must have at least three dimensions (2 given).',
 'dim_shared': 'ValueError: Spectrogram must have at least two dimensions (time and frequency) (1 '
               'given).',
 'iid_no_guard': 'NO_ERROR',
 'p_high': 'ValueError: The value of p must be between 0.0 and 1.0 (1.2 given).',
 'p_low': 'ValueError: The value of p must be between 0.0 and 1.0 (-0.1 given).',
 'width_guard': 'ValueError: Number of columns to be masked should be less than mask_param'}

DEFAULT_TENSOR_VALUE = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

DEFAULT_CENSUS = {'doc_floor': True,
 'iid_dim_guard': True,
 'iid_guard': False,
 'iid_long_split': True,
 'iid_rand_count': 2,
 'iid_repeat': True,
 'iid_transpose': 2,
 'iid_where': True,
 'int_trunc': True,
 'p_default': 1.0,
 'p_shortcut': True,
 'shared_dim_guard': True,
 'shared_guard': True,
 'shared_long_split': True,
 'shared_rand_count': 2,
 'shared_reshape': True,
 'want': ['_get_mask_param', 'mask_along_axis_iid', 'mask_along_axis']}

MODES = ['param', 'rand_calls', 'interval', 'mask', 'output', 'identity', 'errors', 'tensor_value', 'census']

SPEC3 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

SPEC2 = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]

SPEC4 = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
  [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]],
 [[[25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0], [33.0, 34.0, 35.0, 36.0]],
  [[37.0, 38.0, 39.0, 40.0], [41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0]]]]

SPEC1 = [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]]

SPECNEG = [[[-3.0, -2.75, -2.5, -2.25], [-2.0, -1.75, -1.5, -1.25], [-1.0, -0.75, -0.5, -0.25]],
 [[0.0, 0.25, 0.5, 0.75], [1.0, 1.25, 1.5, 1.75], [2.0, 2.25, 2.5, 2.75]]]

SPEC_LEAD23 = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
  [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]],
  [[25.0, 26.0, 27.0, 28.0], [29.0, 30.0, 31.0, 32.0], [33.0, 34.0, 35.0, 36.0]]],
 [[[37.0, 38.0, 39.0, 40.0], [41.0, 42.0, 43.0, 44.0], [45.0, 46.0, 47.0, 48.0]],
  [[49.0, 50.0, 51.0, 52.0], [53.0, 54.0, 55.0, 56.0], [57.0, 58.0, 59.0, 60.0]],
  [[61.0, 62.0, 63.0, 64.0], [65.0, 66.0, 67.0, 68.0], [69.0, 70.0, 71.0, 72.0]]]]

RAND_LEAD23 = [{'dtype': 'float64',
  'shape': [2, 3],
  'values': [[0.31, 0.48, 0.65], [0.8200000000000001, 0.99, 0.16000000000000014]]},
 {'dtype': 'float64',
  'shape': [2, 3],
  'values': [[0.13, 0.42, 0.71], [0.9999999999999999, 0.29000000000000004, 0.5800000000000001]]}]

MASK_LEAD23 = [[[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], [[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]]

OUT_LEAD23 = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
  [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]],
  [[25.0, 0.0, 27.0, 28.0], [29.0, 0.0, 31.0, 32.0], [33.0, 0.0, 35.0, 36.0]]],
 [[[37.0, 0.0, 0.0, 40.0], [41.0, 0.0, 0.0, 44.0], [45.0, 0.0, 0.0, 48.0]],
  [[0.0, 0.0, 51.0, 52.0], [0.0, 0.0, 55.0, 56.0], [0.0, 0.0, 59.0, 60.0]],
  [[61.0, 62.0, 63.0, 64.0], [65.0, 66.0, 67.0, 68.0], [69.0, 70.0, 71.0, 72.0]]]]

OUT_INT_SPEC = [[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]]

PARAM_P1_0 = 10

PARAM_P0_999 = 3

PARAM_P0_9 = 3

PARAM_P0_5 = 2

PARAM_P0_45 = 1

PARAM_P0_39 = 1

PARAM_P0_2 = 0

PARAM_P0_0 = 0

PARAM_PNEG0_3 = -1

PARAM_PNEG0_7 = -2

PARAM_MPNEG2 = -2

PARAM_MP0 = 0

PARAM_MP1 = 1

PARAM_MP2 = 2

PARAM_MP3 = 3

PARAM_MP5 = 5

PARAM_MP10 = 10

RAND_SHARED = [{'dtype': 'float32', 'shape': [1], 'values': [0.699999988079071]},
 {'dtype': 'float32', 'shape': [1], 'values': [0.25]}]

RAND_SHARED_F32 = [{'dtype': 'float32', 'shape': [1], 'values': [0.699999988079071]},
 {'dtype': 'float32', 'shape': [1], 'values': [0.25]}]

RAND_IID = [{'dtype': 'float64', 'shape': [2], 'values': [0.31, 0.48]},
 {'dtype': 'float64', 'shape': [2], 'values': [0.13, 0.42]}]

RAND_IID_F32 = [{'dtype': 'float32', 'shape': [2], 'values': [0.3100000023841858, 0.47999998927116394]},
 {'dtype': 'float32', 'shape': [2], 'values': [0.12999999523162842, 0.41999998688697815]}]

RAND_IID4 = [{'dtype': 'float64', 'shape': [2, 2], 'values': [[0.31, 0.48], [0.65, 0.8200000000000001]]},
 {'dtype': 'float64', 'shape': [2, 2], 'values': [[0.13, 0.42], [0.71, 0.9999999999999999]]}]

RAND_SHARED_EXPLICIT = [{'dtype': 'float32', 'shape': [1], 'values': [0.699999988079071]},
 {'dtype': 'float32', 'shape': [1], 'values': [0.25]}]

RAND_IID_EXPLICIT = [{'dtype': 'float64', 'shape': [2], 'values': [0.7, 0.4]},
 {'dtype': 'float64', 'shape': [2], 'values': [0.25, 0.6]}]

IV_SHARED = {'end': 2, 'min_value': 0.4750000238418579, 'start': 0, 'value': 2.0999999046325684, 'width': 2}

IV_IID = [{'end': 2, 'min_value': 0.4750000000000001, 'start': 0, 'value': 2.0999999999999996, 'width': 2},
 {'end': 2, 'min_value': 1.68, 'start': 1, 'value': 1.2000000000000002, 'width': 1}]

IV_BIG = {'end': 7, 'min_value': -0.75, 'start': 0, 'value': 7.0, 'width': 7}

IV_GUARD = {'end': 4, 'min_value': -0.125, 'start': 0, 'value': 4.5, 'width': 4}

IV_NO_GUARD = [{'end': 4, 'min_value': -0.125, 'start': 0, 'value': 4.5, 'width': 4},
 {'end': 2, 'min_value': 1.68, 'start': 1, 'value': 1.2000000000000002, 'width': 1}]

IV_F32 = [{'end': 2, 'min_value': 0.4750000238418579, 'start': 0, 'value': 2.0999999046325684, 'width': 2},
 {'end': 2, 'min_value': 1.6800000667572021, 'start': 1, 'value': 1.2000000476837158, 'width': 1}]

IV_TRUNC = [{'end': 3, 'min_value': 0.029499999999999894, 'start': 0, 'value': 3.95, 'width': 3},
 {'end': 4, 'min_value': -0.9405000000000001, 'start': 0, 'value': 4.95, 'width': 4}]

IV_FREQ_P05 = [{'end': 0, 'min_value': 0.3497, 'start': 0, 'value': 0.31, 'width': 0},
 {'end': 1, 'min_value': 1.0584, 'start': 1, 'value': 0.48, 'width': 0}]

MASK_FREQ_P05 = [[0, 0, 0], [0, 0, 0]]

OUT_FREQ_P05 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

IV_SPLIT = {'end': 2, 'min_value': 0.800000011920929, 'start': 0, 'value': 2.700000047683716, 'width': 2}

MASK_SPLIT = [1, 1, 0, 0]

OUT_SPLIT = [[[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]],
 [[0.0, 0.0, 15.0, 16.0], [0.0, 0.0, 19.0, 20.0], [0.0, 0.0, 23.0, 24.0]]]

ERR_GUARD_EQ = {'msg': 'Number of columns to be masked should be less than mask_param', 'type': 'ValueError'}

MASK_SHARED = [1, 1, 0, 0]

MASK_IID = [[1, 1, 0, 0], [0, 1, 0, 0]]

MASK_FREQ_SHARED = [1, 1, 0]

MASK_FREQ_IID = [[1, 1, 0], [0, 1, 0]]

MASK_BIG = [1, 1, 1, 1]

MASK_ZERO = [0, 0, 0, 0]

MASK_MP1 = [0, 0, 0, 0]

MASK_IID4 = [[[0, 0, 0, 0], [0, 1, 0, 0]], [[0, 1, 0, 0], [0, 1, 1, 0]]]

MASK_SINGLE_T = [1]

OUT_SHARED = [[[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]],
 [[0.0, 0.0, 15.0, 16.0], [0.0, 0.0, 19.0, 20.0], [0.0, 0.0, 23.0, 24.0]]]

OUT_SHARED_FREQ = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [9.0, 10.0, 11.0, 12.0]],
 [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_SHARED_FREQ_NEG = [[[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [9.0, 10.0, 11.0, 12.0]],
 [[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_INT_VALUE = [[[7.0, 7.0, 3.0, 4.0], [7.0, 7.0, 7.0, 8.0], [7.0, 7.0, 11.0, 12.0]],
 [[13.0, 7.0, 15.0, 16.0], [17.0, 7.0, 19.0, 20.0], [21.0, 7.0, 23.0, 24.0]]]

EMPTY_SPEC_SHARED = {'msg': 'Spectrogram must have at least two dimensions (time and frequency) (1 given).',
 'type': 'ValueError'}

EMPTY_SPEC_IID = {'msg': 'Spectrogram must have at least three dimensions (1 given).', 'type': 'ValueError'}

EMPTY_DRAWS = {'msg': 'draw size 0 does not match torch.rand shape (2,)', 'type': 'ValueError'}

OUT_IID = [[[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

OUT_IID_FREQ = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [0.0, 0.0, 0.0, 0.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_DEFAULT = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

OUT_2D = [[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]]

OUT_4D = [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
  [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]],
 [[[25.0, 0.0, 27.0, 28.0], [29.0, 0.0, 31.0, 32.0], [33.0, 0.0, 35.0, 36.0]],
  [[37.0, 0.0, 0.0, 40.0], [41.0, 0.0, 0.0, 44.0], [45.0, 0.0, 0.0, 48.0]]]]

OUT_4D_SHARED = [[[[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]],
  [[0.0, 0.0, 15.0, 16.0], [0.0, 0.0, 19.0, 20.0], [0.0, 0.0, 23.0, 24.0]]],
 [[[0.0, 0.0, 27.0, 28.0], [0.0, 0.0, 31.0, 32.0], [0.0, 0.0, 35.0, 36.0]],
  [[0.0, 0.0, 39.0, 40.0], [0.0, 0.0, 43.0, 44.0], [0.0, 0.0, 47.0, 48.0]]]]

OUT_F32 = [[[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 7.0, 8.0], [0.0, 0.0, 11.0, 12.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

OUT_NEG_VALUE = [[[-1.0, -1.0, 3.0, 4.0], [-1.0, -1.0, 7.0, 8.0], [-1.0, -1.0, 11.0, 12.0]],
 [[13.0, -1.0, 15.0, 16.0], [17.0, -1.0, 19.0, 20.0], [21.0, -1.0, 23.0, 24.0]]]

OUT_TENSOR = [[[-1.0, -1.0, 3.0, 4.0], [-1.0, -1.0, 7.0, 8.0], [-1.0, -1.0, 11.0, 12.0]],
 [[13.0, -1.0, 15.0, 16.0], [17.0, -1.0, 19.0, 20.0], [21.0, -1.0, 23.0, 24.0]]]

OUT_BIG = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
 [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]

OUT_P039 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_P0 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_MP0 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_MP1 = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
 [[13.0, 14.0, 15.0, 16.0], [17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0]]]

OUT_NO_GUARD = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
 [[13.0, 0.0, 15.0, 16.0], [17.0, 0.0, 19.0, 20.0], [21.0, 0.0, 23.0, 24.0]]]

OUT_NEG_SPEC = [[[0.0, 0.0, -2.5, -2.25], [0.0, 0.0, -1.5, -1.25], [0.0, 0.0, -0.5, -0.25]],
 [[0.0, 0.0, 0.5, 0.75], [1.0, 0.0, 1.5, 1.75], [2.0, 0.0, 2.5, 2.75]]]

OUT_SINGLE_T = [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]

IDENT_MP0_P1_0 = {'err': None, 'rand_calls': 0, 'same_object': True}

IDENT_MPNEG3_P1_0 = {'err': None, 'rand_calls': 0, 'same_object': True}

IDENT_MP10_P0_0 = {'err': None, 'rand_calls': 0, 'same_object': True}

IDENT_MP1_P1_0 = {'err': None, 'rand_calls': 2, 'same_object': False}

IDENT_MP3_P1_0 = {'err': None, 'rand_calls': 2, 'same_object': False}

IDENT_ERR_P_HIGH = {'err': {'msg': 'The value of p must be between 0.0 and 1.0 (1.2 given).', 'type': 'ValueError'},
 'rand_calls': 0,
 'same_object': False}

IDENT_ERR_AXIS = {'err': {'msg': 'Only Frequency and Time masking are supported (axis 1 and axis 2 supported; 0 given).',
         'type': 'ValueError'},
 'rand_calls': 0,
 'same_object': False}

ERRORS = {'axis_negative': 'ValueError: Only Frequency and Time masking are supported (axis 1 and axis 2 '
                  'supported; -1 given).',
 'axis_oob': 'ValueError: Only Frequency and Time masking are supported (axis 1 and axis 2 supported; '
             '0 given).',
 'dim_iid': 'ValueError: Spectrogram must have at least three dimensions (2 given).',
 'dim_shared': 'ValueError: Spectrogram must have at least two dimensions (time and frequency) (1 '
               'given).',
 'iid_no_guard': 'NO_ERROR',
 'p_high': 'ValueError: The value of p must be between 0.0 and 1.0 (1.2 given).',
 'p_low': 'ValueError: The value of p must be between 0.0 and 1.0 (-0.1 given).',
 'width_guard': 'ValueError: Number of columns to be masked should be less than mask_param'}

ERR_AXIS_OOB = {'msg': 'Only Frequency and Time masking are supported (axis 1 and axis 2 supported; 0 given).',
 'type': 'ValueError'}

ERR_AXIS_NEG = {'msg': 'Only Frequency and Time masking are supported (axis 1 and axis 2 supported; -1 given).',
 'type': 'ValueError'}

ERR_P_HIGH = {'msg': 'The value of p must be between 0.0 and 1.0 (1.2 given).', 'type': 'ValueError'}

ERR_P_LOW = {'msg': 'The value of p must be between 0.0 and 1.0 (-0.1 given).', 'type': 'ValueError'}

ERR_DIM_SHARED = {'msg': 'Spectrogram must have at least two dimensions (time and frequency) (1 given).',
 'type': 'ValueError'}

ERR_DIM_IID = {'msg': 'Spectrogram must have at least three dimensions (2 given).', 'type': 'ValueError'}

ERR_GUARD = {'msg': 'Number of columns to be masked should be less than mask_param', 'type': 'ValueError'}

ERR_DRAWS = {'msg': 'draw size 3 does not match torch.rand shape (2,)', 'type': 'ValueError'}

ERR_TENSOR_MULTI = {'msg': 'The size of tensor a (4) must match the size of tensor b (8) at non-singleton dimension 2',
 'type': 'RuntimeError'}

CENSUS = {'doc_floor': True,
 'iid_dim_guard': True,
 'iid_guard': False,
 'iid_long_split': True,
 'iid_rand_count': 2,
 'iid_repeat': True,
 'iid_transpose': 2,
 'iid_where': True,
 'int_trunc': True,
 'p_default': 1.0,
 'p_shortcut': True,
 'shared_dim_guard': True,
 'shared_guard': True,
 'shared_long_split': True,
 'shared_rand_count': 2,
 'shared_reshape': True,
 'want': ['_get_mask_param', 'mask_along_axis_iid', 'mask_along_axis']}

import math


def _near(got, want, tag, rtol=1e-8, atol=1e-9):
    if isinstance(want, (list, tuple)):
        assert isinstance(got, (list, tuple)), tag + ": expected a list"
        assert len(got) == len(want), "%s: length %d != %d" % (tag, len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            _near(a, b, "%s[%d]" % (tag, i), rtol, atol)
        return
    if isinstance(want, dict):
        assert isinstance(got, dict), tag + ": expected a dict"
        assert sorted(got) == sorted(want), "%s: keys %r != %r" % (tag, sorted(got), sorted(want))
        for k in want:
            _near(got[k], want[k], "%s.%s" % (tag, k), rtol, atol)
        return
    if isinstance(want, bool) or isinstance(got, bool):
        assert got is want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, str) or want is None:
        assert got == want, "%s: %r != %r" % (tag, got, want)
        return
    if isinstance(want, float) and math.isnan(want):
        assert isinstance(got, float) and math.isnan(got), "%s: %r != nan" % (tag, got)
        return
    assert abs(float(got) - float(want)) <= atol + rtol * abs(float(want)), \
        "%s: %r != %r" % (tag, got, want)


def _raises(fn, tag, needle, excs=(ValueError, TypeError)):
    try:
        fn()
    except excs as e:
        assert needle in str(e), "%s: wrong message %r (want %r)" % (tag, str(e), needle)
    except BaseException as e:
        raise AssertionError("%s: wrong exception %s %s" % (tag, type(e).__name__, e))
    else:
        raise AssertionError("%s: no exception" % tag)


def C(solve, mode, **kw):
    return solve(mode, **kw)


def _err(res, tag, typ, needle):
    assert isinstance(res, dict), tag + ": expected an error dict"
    assert res.get("type") == typ, "%s: type %r != %r" % (tag, res.get("type"), typ)
    assert needle in res.get("msg", ""), "%s: message %r missing %r" % (tag, res.get("msg"), needle)


# ------------------------------------------------------------- default modes
def check_default_param(solve):
    _near(C(solve, "param"), DEFAULT_PARAM, "default param")


def check_default_rand_calls(solve):
    _near(C(solve, "rand_calls"), DEFAULT_RAND_CALLS, "default rand_calls")


def check_default_interval(solve):
    _near(C(solve, "interval"), DEFAULT_INTERVAL, "default interval")


def check_default_mask(solve):
    _near(C(solve, "mask"), DEFAULT_MASK, "default mask")


def check_default_output(solve):
    _near(C(solve, "output"), DEFAULT_OUTPUT, "default output")


def check_default_identity(solve):
    _near(C(solve, "identity"), DEFAULT_IDENTITY, "default identity")


def check_default_errors(solve):
    _near(C(solve, "errors"), DEFAULT_ERRORS, "default errors")


def check_default_tensor_value(solve):
    _near(C(solve, "tensor_value"), DEFAULT_TENSOR_VALUE, "default tensor_value")


def check_default_census(solve):
    _near(C(solve, "census"), DEFAULT_CENSUS, "default census")


def check_modes_tuple(solve):
    _near(MODES, ["param", "rand_calls", "interval", "mask", "output",
                  "identity", "errors", "tensor_value", "census"], "MODES")


# ------------------------------------------------------------------- param
def check_param_p_shortcut(solve):
    _near(C(solve, "param", mask_param=10, p=1.0), PARAM_P1_0, "p shortcut")


def check_param_p_trunc(solve):
    _near(C(solve, "param", mask_param=10, p=0.999), PARAM_P0_999, "p 0.999 trunc")
    _near(C(solve, "param", mask_param=10, p=0.9), PARAM_P0_9, "p 0.9 trunc")
    _near(C(solve, "param", mask_param=10, p=0.45), PARAM_P0_45, "p 0.45 trunc")
    _near(C(solve, "param", mask_param=10, p=0.39), PARAM_P0_39, "p 0.39 trunc")


def check_param_p_half(solve):
    _near(C(solve, "param", mask_param=10, p=0.5), PARAM_P0_5, "p 0.5")


def check_param_p_small(solve):
    _near(C(solve, "param", mask_param=10, p=0.2), PARAM_P0_2, "p 0.2")
    _near(C(solve, "param", mask_param=10, p=0.0), PARAM_P0_0, "p 0")


def check_param_mp_census(solve):
    for mp, key in [(-2, "NEG2"), (0, "0"), (1, "1"), (2, "2"),
                    (3, "3"), (5, "5"), (10, "10")]:
        _near(C(solve, "param", mask_param=mp), globals()["PARAM_MP" + key],
              "param mp=%d" % mp)


def check_param_p_scales_axis(solve):
    _near(C(solve, "param", mask_param=9, p=0.5, axis=1), 1,
          "int(3*0.5)=1 on freq axis")
    _near(C(solve, "param", mask_param=9, p=0.5, axis=2), 2,
          "int(4*0.5)=2 on time axis")


def check_param_float_mask_param(solve):
    _near(C(solve, "param", mask_param=2.75, p=0.5), 2.0,
          "mask_param is not coerced to int by _get_mask_param")


# --------------------------------------------------------------- rand_calls
def check_rand_shared(solve):
    _near(C(solve, "rand_calls", variant="shared"), RAND_SHARED, "rand shared")


def check_rand_shared_f32(solve):
    _near(C(solve, "rand_calls", variant="shared", dtype="float32"),
          RAND_SHARED_F32, "rand shared float32")


def check_rand_iid(solve):
    _near(C(solve, "rand_calls", variant="iid"), RAND_IID, "rand iid")


def check_rand_iid_f32(solve):
    _near(C(solve, "rand_calls", variant="iid", dtype="float32"),
          RAND_IID_F32, "rand iid float32")


def check_rand_iid_4d(solve):
    _near(C(solve, "rand_calls", spec=SPEC4), RAND_IID4, "rand iid 4D")


def check_rand_explicit(solve):
    _near(C(solve, "rand_calls", variant="shared", value_draws=[0.7],
            start_draws=[0.25]), RAND_SHARED_EXPLICIT, "rand shared explicit")
    _near(C(solve, "rand_calls", variant="iid", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), RAND_IID_EXPLICIT, "rand iid explicit")


def check_rand_two_calls_distinct(solve):
    r = C(solve, "rand_calls")
    assert len(r) == 2, "two torch.rand calls"
    assert r[0]["values"] != r[1]["values"], "value and start draws differ"
    assert r[0]["shape"] == r[1]["shape"], "draw shapes agree"


def check_rand_shared_dtype_ignores_spec(solve):
    for dt in ("float64", "float32"):
        r = C(solve, "rand_calls", variant="shared", dtype=dt)
        assert r[0]["dtype"] == "float32", "shared rand stays float32 for " + dt


def check_rand_identity_has_none(solve):
    _near(C(solve, "rand_calls", mask_param=0), [], "identity has no draws")


def check_rand_bad_shape(solve):
    r = C(solve, "rand_calls", value_draws=[0.1, 0.2, 0.3],
          start_draws=[0.1, 0.2, 0.3])
    _near(r, [], "bad draws are rejected before recording")


# ---------------------------------------------------------------- interval
def check_interval_shared(solve):
    _near(C(solve, "interval", variant="shared", value_draws=[0.7],
            start_draws=[0.25]), IV_SHARED, "interval shared")


def check_interval_iid(solve):
    _near(C(solve, "interval", variant="iid", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), IV_IID, "interval iid")


def check_interval_big(solve):
    _near(C(solve, "interval", variant="shared", mask_param=10,
            value_draws=[0.7], start_draws=[0.25]), IV_BIG, "interval big")


def check_interval_guard(solve):
    _near(C(solve, "interval", variant="shared", mask_param=3,
            value_draws=[1.5], start_draws=[0.25]), IV_GUARD, "interval guard")


def check_interval_no_guard(solve):
    _near(C(solve, "interval", mask_param=3, value_draws=[1.5, 0.4],
            start_draws=[0.25, 0.6]), IV_NO_GUARD, "interval no guard")


def check_interval_f32(solve):
    _near(C(solve, "interval", dtype="float32", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), IV_F32, "interval float32")


def check_interval_trunc(solve):
    _near(C(solve, "interval", mask_param=5, value_draws=[0.79, 0.99],
            start_draws=[0.59, 0.99]), IV_TRUNC, "interval trunc grid")


def check_interval_separate_long(solve):
    _near(C(solve, "interval", variant="shared", mask_param=10,
            value_draws=[0.27], start_draws=[0.6153846153846154]),
          IV_SPLIT, "interval separate .long()")
    assert IV_SPLIT["end"] == 2, "fixture pins end=2 (not trunc(min+value)=3)"
    assert IV_SPLIT["min_value"] + IV_SPLIT["value"] > 3.0, \
        "fixture has fractional parts summing above 1"


def check_interval_width_invariant(solve):
    for cfg in (dict(variant="shared", value_draws=[0.7], start_draws=[0.25]),
                dict(value_draws=[0.7, 0.4], start_draws=[0.25, 0.6])):
        iv = C(solve, "interval", **cfg)
        nodes = [iv] if isinstance(iv, dict) else iv
        for node in nodes:
            assert node["width"] == node["end"] - node["start"], "width = end - start"
            assert node["end"] == node["start"] + math.trunc(node["value"]), \
                "end uses trunc(value)"


def check_interval_identity_none(solve):
    assert C(solve, "interval", mask_param=0) is None, "identity has no interval"


def check_interval_min_value_negative(solve):
    _near(IV_BIG["min_value"], -0.75, "negative min_value")
    _near(IV_BIG["start"], 0, "negative min_value truncates toward zero")


# -------------------------------------------------------------------- mask
def check_mask_shared(solve):
    _near(C(solve, "mask", variant="shared", value_draws=[0.7],
            start_draws=[0.25]), MASK_SHARED, "mask shared")


def check_mask_iid(solve):
    _near(C(solve, "mask", variant="iid", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), MASK_IID, "mask iid")


def check_mask_freq_shared(solve):
    _near(C(solve, "mask", variant="shared", axis=1, value_draws=[0.7],
            start_draws=[0.25]), MASK_FREQ_SHARED, "mask freq shared")


def check_mask_freq_iid(solve):
    _near(C(solve, "mask", variant="iid", axis=1, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), MASK_FREQ_IID, "mask freq iid")


def check_mask_big(solve):
    _near(C(solve, "mask", variant="shared", mask_param=10,
            value_draws=[0.7], start_draws=[0.25]), MASK_BIG, "mask big")


def check_mask_zero(solve):
    _near(C(solve, "mask", variant="shared", value_draws=[0.0],
            start_draws=[0.5]), MASK_ZERO, "mask zero width")


def check_mask_mp1(solve):
    _near(C(solve, "mask", variant="shared", mask_param=1,
            value_draws=[0.9], start_draws=[0.5]), MASK_MP1, "mask mp=1")


def check_mask_iid_4d(solve):
    _near(C(solve, "mask", spec=SPEC4), MASK_IID4, "mask iid 4D nesting")


def check_mask_single_t(solve):
    _near(C(solve, "mask", variant="shared", spec=SPEC1,
            value_draws=[0.7], start_draws=[0.25]), MASK_SINGLE_T,
          "mask single time step")


def check_mask_split(solve):
    _near(C(solve, "mask", variant="shared", mask_param=10,
            value_draws=[0.27], start_draws=[0.6153846153846154]),
          MASK_SPLIT, "mask separate truncation")


def check_mask_half_open(solve):
    assert MASK_SHARED == [1, 1, 0, 0], "half-open [start,end)"
    assert MASK_IID[1] == [0, 1, 0, 0], "start is inclusive, end exclusive"


def check_mask_identity_empty(solve):
    _near(C(solve, "mask", mask_param=0), [], "identity has no mask rows")


# ------------------------------------------------------------------ output
def check_output_shared(solve):
    _near(C(solve, "output", variant="shared", value_draws=[0.7],
            start_draws=[0.25]), OUT_SHARED, "output shared")


def check_output_shared_freq(solve):
    _near(C(solve, "output", variant="shared", axis=1,
            value_draws=[0.7], start_draws=[0.25]), OUT_SHARED_FREQ,
          "output shared freq")


def check_output_iid(solve):
    _near(C(solve, "output", variant="iid", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_IID, "output iid")


def check_output_iid_freq(solve):
    _near(C(solve, "output", variant="iid", axis=1, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_IID_FREQ, "output iid freq")


def check_output_default(solve):
    _near(C(solve, "output"), OUT_DEFAULT, "output default")


def check_output_2d(solve):
    _near(C(solve, "output", variant="shared", spec=SPEC2,
            value_draws=[0.7], start_draws=[0.25]), OUT_2D, "output 2D")


def check_output_4d(solve):
    _near(C(solve, "output", spec=SPEC4, axis=3), OUT_4D, "output 4D iid")


def check_output_4d_shared(solve):
    _near(C(solve, "output", variant="shared", spec=SPEC4, axis=3,
            value_draws=[0.7], start_draws=[0.25]), OUT_4D_SHARED,
          "output 4D shared pack")


def check_output_f32(solve):
    _near(C(solve, "output", dtype="float32", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_F32, "output float32")


def check_output_neg_value(solve):
    _near(C(solve, "output", mask_value=-1.0, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_NEG_VALUE, "output negative fill")


def check_output_tensor(solve):
    _near(C(solve, "tensor_value", mask_value=-1.0, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_TENSOR, "tensor_value output")


def check_output_big(solve):
    _near(C(solve, "output", variant="shared", mask_param=10,
            value_draws=[0.7], start_draws=[0.25]), OUT_BIG, "output big")


def check_output_p039(solve):
    _near(C(solve, "output", p=0.39, mask_param=10), OUT_P039, "output p 0.39")


def check_output_p0(solve):
    _near(C(solve, "output", p=0.0, mask_param=10), OUT_P0, "output p 0")


def check_output_mp0(solve):
    _near(C(solve, "output", mask_param=0), OUT_MP0, "output mp 0")
    _near(OUT_MP0, SPEC3, "mp 0 leaves spectrum unchanged")


def check_output_mp1(solve):
    _near(C(solve, "output", mask_param=1), OUT_MP1, "output mp 1")


def check_output_no_guard(solve):
    _near(C(solve, "output", mask_param=3, value_draws=[1.5, 0.4],
            start_draws=[0.25, 0.6]), OUT_NO_GUARD, "iid no width guard")


def check_output_neg_spec(solve):
    _near(C(solve, "output", spec=SPECNEG, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), OUT_NEG_SPEC,
          "output on a negative / fractional spectrum")


def check_output_single_t(solve):
    _near(C(solve, "output", variant="shared", spec=SPEC1,
            value_draws=[0.7], start_draws=[0.25]), OUT_SINGLE_T,
          "output single time step")


def check_output_split(solve):
    _near(C(solve, "output", variant="shared", mask_param=10,
            value_draws=[0.27], start_draws=[0.6153846153846154],
            mask_value=0.0), OUT_SPLIT, "output separate truncation")


def check_output_uniform_draws_agree(solve):
    a = C(solve, "output", variant="iid", value_draws=[0.7, 0.7],
          start_draws=[0.25, 0.25], mask_value=0.0)
    b = C(solve, "output", variant="shared", value_draws=[0.7],
          start_draws=[0.25], mask_value=0.0)
    _near(a, b, "uniform iid draws reproduce shared masks")


def check_output_shape_preserved(solve):
    for cfg, want in [({}, [2, 3, 4]),
                      (dict(spec=SPEC2, variant="shared"), [3, 4]),
                      (dict(spec=SPEC4, axis=3), [2, 2, 3, 4])]:
        def shape(x):
            return [] if not isinstance(x, list) else [len(x)] + shape(x[0])
        _near(shape(C(solve, "output", **cfg)), want, "shape preserved")


def check_output_distinct_rows(solve):
    # Every injected row must be different: mutations that ignore the leading
    # index or reverse the pack order must be caught here.
    assert OUT_IID[0][0][0] != OUT_IID[1][0][0], "rows differ"
    assert MASK_IID[0] != MASK_IID[1], "iid rows differ"
    assert OUT_4D[0][0][0][0] != OUT_4D[1][0][0][0], "4D rows differ"


# ---------------------------------------------------------------- identity
def check_identity_mp0(solve):
    _near(C(solve, "identity", mask_param=0), IDENT_MP0_P1_0, "identity mp0")


def check_identity_negative(solve):
    _near(C(solve, "identity", mask_param=-3), IDENT_MPNEG3_P1_0,
          "identity negative mp")


def check_identity_p0(solve):
    _near(C(solve, "identity", mask_param=10, p=0.0), IDENT_MP10_P0_0,
          "identity p0")


def check_identity_mp1_not_shortcut(solve):
    _near(C(solve, "identity", mask_param=1), IDENT_MP1_P1_0,
          "mp1 is not the identity shortcut")


def check_identity_mp3_not_shortcut(solve):
    _near(C(solve, "identity", mask_param=3), IDENT_MP3_P1_0,
          "mp3 is not the identity shortcut")


def check_identity_no_draws(solve):
    r = C(solve, "identity", mask_param=0)
    assert r["rand_calls"] == 0, "shortcut consumes no draw"
    assert r["same_object"] is True, "shortcut returns the original object"
    assert r["err"] is None, "shortcut has no error"


# ------------------------------------------------------------------ errors
def check_errors_dict(solve):
    _near(C(solve, "errors"), ERRORS, "errors dict")


def check_error_dim_shared(solve):
    _err(C(solve, "output", variant="shared", spec=[1.0, 2.0, 3.0],
           axis=0, mask_param=2), "dim shared", "ValueError",
         "at least two dimensions")


def check_error_dim_iid(solve):
    _err(C(solve, "output", spec=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
           axis=1, mask_param=2), "dim iid", "ValueError",
         "at least three dimensions")


def check_error_axis_oob(solve):
    _near(C(solve, "output", axis=0, mask_param=2), ERR_AXIS_OOB, "axis oob")


def check_error_axis_negative(solve):
    _near(C(solve, "output", axis=-1, mask_param=2), ERR_AXIS_NEG,
          "negative axis")


def check_error_p_high(solve):
    _near(C(solve, "output", p=1.2, mask_param=2), ERR_P_HIGH, "p high")


def check_error_p_low(solve):
    _near(C(solve, "output", p=-0.1, mask_param=2), ERR_P_LOW, "p low")


def check_error_guard(solve):
    _near(C(solve, "output", variant="shared", mask_param=3,
            value_draws=[1.5], start_draws=[0.25]), ERR_GUARD, "width guard")


def check_error_guard_equal(solve):
    _near(C(solve, "output", variant="shared", mask_param=3,
            value_draws=[1.0], start_draws=[0.25]), ERR_GUARD_EQ,
          "width guard fires at width == mask_param")


def check_error_draws(solve):
    _near(C(solve, "output", value_draws=[0.1, 0.2, 0.3],
            start_draws=[0.1, 0.2, 0.3]), ERR_DRAWS, "draw shape mismatch")


def check_error_p_boundary_ok(solve):
    for p in (0.0, 1.0):
        assert isinstance(C(solve, "output", p=p, mask_param=2), list), \
            "p=%s accepted" % p


def check_error_guard_only_shared(solve):
    assert ERRORS["iid_no_guard"] == "NO_ERROR", "iid has no width guard"
    assert "should be less than mask_param" in ERRORS["width_guard"], \
        "shared has width guard"


def check_error_messages_verbatim(solve):
    e = C(solve, "errors")
    assert "(1 given)" in e["dim_shared"], "dim shared reports dim"
    assert "(2 given)" in e["dim_iid"], "dim iid reports dim"
    assert "axis 1 and axis 2 supported; 0 given" in e["axis_oob"], "axis text"
    assert "-1 given" in e["axis_negative"], "negative axis text"
    assert "(1.2 given)" in e["p_high"] and "(-0.1 given)" in e["p_low"], "p text"


# ------------------------------------------------------------ tensor_value
def check_tensor_value_scalar(solve):
    _near(C(solve, "tensor_value", mask_value=-1.0,
            value_draws=[0.7, 0.4], start_draws=[0.25, 0.6]),
          OUT_TENSOR, "tensor scalar fill")


def check_tensor_value_default(solve):
    _near(C(solve, "tensor_value"), DEFAULT_TENSOR_VALUE, "tensor default")


def check_tensor_value_equals_float(solve):
    _near(C(solve, "tensor_value", mask_value=-1.0,
            value_draws=[0.7, 0.4], start_draws=[0.25, 0.6]),
          C(solve, "output", mask_value=-1.0, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6]), "tensor and float branch agree")


def check_tensor_value_forces_iid(solve):
    _near(C(solve, "tensor_value", variant="shared", mask_value=0.0),
          C(solve, "output", variant="iid", mask_value=0.0),
          "tensor_value always uses the iid function")


def check_tensor_value_multi_error(solve):
    _err(C(solve, "tensor_value", mask_value=[-1.0, -2.0],
           value_draws=[0.7, 0.4], start_draws=[0.25, 0.6]),
         "multi tensor", "RuntimeError", "non-scalar")


def check_tensor_value_zero(solve):
    _near(C(solve, "tensor_value", mask_value=0.0),
          C(solve, "output", mask_value=0.0), "tensor zero equals float zero")


# ------------------------------------------------------------------ census
def check_census_dict(solve):
    _near(C(solve, "census"), CENSUS, "census dict")


def check_census_want(solve):
    _near(CENSUS["want"], ["_get_mask_param", "mask_along_axis_iid",
                           "mask_along_axis"], "census want")


def check_census_guards(solve):
    c = C(solve, "census")
    assert c["shared_guard"] is True and c["iid_guard"] is False, "guard split"
    assert c["p_shortcut"] is True and c["int_trunc"] is True, "param quirks"
    assert c["doc_floor"] is True, "docstring says floor"


def check_census_counts(solve):
    c = C(solve, "census")
    assert c["shared_rand_count"] == 2 and c["iid_rand_count"] == 2, "rand count"
    assert c["iid_transpose"] == 2, "two transposes"
    assert c["p_default"] == 1.0, "p default"


def check_census_flags(solve):
    c = C(solve, "census")
    for k in ("iid_repeat", "iid_where", "shared_dim_guard", "iid_dim_guard",
              "shared_reshape", "iid_long_split", "shared_long_split"):
        assert c[k] is True, "census flag " + k


# --------------------------------------------------------------- hardening
def check_unknown_mode(solve):
    _raises(lambda: solve("evd", psd_s=1), "unknown mode", "unknown mode")
    try:
        solve("evd")
    except (ValueError, TypeError) as e:
        msg = str(e)
    else:
        raise AssertionError("unknown mode must raise")
    for name in MODES:
        assert name in msg, "unknown-mode message must list " + name


def check_modes_message_exact(solve):
    try:
        solve("bogus")
    except ValueError as e:
        assert "expected one of" in str(e), "message names the mode list"
        assert "tensor_value" in str(e) and "census" in str(e), \
            "message lists later modes too"


def check_spec_ragged(solve):
    _raises(lambda: C(solve, "output", spec=[[1.0, 2.0], [3.0]],
                      variant="shared"), "ragged spec", "rectangular")


def check_spec_non_numeric(solve):
    _raises(lambda: C(solve, "output", spec=[[1.0, "x"], [3.0, 4.0]],
                      variant="shared"), "string leaf", "must contain numbers")


def check_spec_bool_leaf(solve):
    _raises(lambda: C(solve, "output", spec=[[1.0, True], [3.0, 4.0]],
                      variant="shared"), "bool leaf", "must contain numbers")


def check_spec_tuple_input(solve):
    sp = tuple(tuple(float(i * 4 + j + 1) for j in range(4)) for i in range(3))
    _near(C(solve, "output", spec=sp, variant="shared",
            value_draws=[0.7], start_draws=[0.25]),
          OUT_2D, "tuple spec accepted")


def check_draws_tuple_input(solve):
    _near(C(solve, "output", value_draws=(0.7, 0.4),
            start_draws=(0.25, 0.6)), OUT_IID, "tuple draws accepted")


def check_draws_non_numeric(solve):
    _raises(lambda: C(solve, "output", value_draws=[0.7, "x"],
                      start_draws=[0.25, 0.6]), "draw leaf", "must contain numbers")


def check_draws_bool_leaf(solve):
    _raises(lambda: C(solve, "output", value_draws=[True, 0.4],
                      start_draws=[0.25, 0.6]), "bool draw", "must contain numbers")


def check_start_draws_shape(solve):
    _near(C(solve, "output", value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6, 0.9]), ERR_DRAWS,
          "start draw mismatch uses the same message")


def check_variant_fallback(solve):
    # any variant string other than "iid" dispatches to mask_along_axis
    _near(C(solve, "output", variant="bogus", spec=SPEC3,
            value_draws=[0.7], start_draws=[0.25]),
          C(solve, "output", variant="shared", spec=SPEC3,
            value_draws=[0.7], start_draws=[0.25]), "variant fallback")


def check_axis_negative_rejected(solve):
    _err(C(solve, "output", axis=-1), "axis -1", "ValueError",
         "-1 given")


def check_axis_too_large(solve):
    _err(C(solve, "output", axis=3), "axis 3", "ValueError",
         "Only Frequency and Time masking")


def check_p_nan_rejected(solve):
    _err(C(solve, "output", p=float("nan")), "p nan", "ValueError",
         "The value of p must be between")


def check_dtype_census(solve):
    for dt, want in (("float64", "float64"), ("float32", "float32")):
        r = C(solve, "rand_calls", dtype=dt)
        assert r[0]["dtype"] == want, "dtype " + dt


def check_mask_value_float(solve):
    _near(C(solve, "output", mask_value=7.25, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6])[1][0][1], 7.25, "float fill value")


def check_mask_value_zero_is_not_identity(solve):
    assert C(solve, "identity", mask_param=3)["same_object"] is False, \
        "mask_param>=1 always allocates a masked copy"


def check_4d_draw_nesting(solve):
    r = C(solve, "rand_calls", spec=SPEC4)
    assert r[0]["shape"] == [2, 2], "4D rand shape"
    assert isinstance(r[0]["values"][0], list), "4D draw nesting"


def check_4d_mask_nesting(solve):
    _near(C(solve, "mask", spec=SPEC4), MASK_IID4, "4D mask nesting")


def check_output_shared_same_mask_all_examples(solve):
    out = C(solve, "output", variant="shared", spec=SPEC4, axis=3,
            value_draws=[0.7], start_draws=[0.25])
    flags = lambda m: [v == 0.0 for row in m for v in row]
    for lead in out:
        for m in lead:
            assert flags(m) == flags(out[0][0]), "shared mask across all examples"


def check_identity_output_is_input(solve):
    _near(C(solve, "output", mask_param=0), SPEC3, "identity output equals input")


def check_output_f32_shared(solve):
    _near(C(solve, "output", dtype="float32", variant="shared",
            value_draws=[0.7], start_draws=[0.25]),
          C(solve, "output", dtype="float64", variant="shared",
            value_draws=[0.7], start_draws=[0.25]),
          "shared interval dtype does not change the mask")


def check_default_spec_shape(solve):
    def shape(x):
        return [] if not isinstance(x, list) else [len(x)] + shape(x[0])
    _near(shape(SPEC3), [2, 3, 4], "SPEC3 shape")
    _near(shape(SPEC4), [2, 2, 3, 4], "SPEC4 shape")
    _near(shape(SPEC1), [2, 3, 1], "SPEC1 shape")


def check_grid_consistency(solve):
    for sh, sp in [((2, 3, 4), SPEC3), ((2, 2, 3, 4), SPEC4)]:
        for variant in ("shared", "iid"):
            for axis in (len(sh) - 2, len(sh) - 1):
                for mp in (1, 3, 6):
                    kw = dict(spec=sp, variant=variant, axis=axis,
                              mask_param=mp)
                    iv = C(solve, "interval", **kw)
                    rows = C(solve, "mask", **kw)
                    out = C(solve, "output", **kw)
                    assert isinstance(out, list), "grid output"
                    assert (iv is None) == (rows == []), "grid interval/mask"


# ------------------------------------------------- second-round hardening
def check_unknown_mode_exact_order(solve):
    # The mode list in the error message is the solution's own MODES tuple,
    # joined in order -- checking mere membership would miss a reordering.
    try:
        solve("bogus")
    except (ValueError, TypeError) as e:
        msg = str(e)
    else:
        raise AssertionError("unknown mode must raise")
    want = "(expected one of " + ", ".join(MODES) + ")"
    assert want in msg, "mode list must be joined in order: %r" % msg


def check_identity_error_forwarded(solve):
    _near(C(solve, "identity", p=1.2, mask_param=2), IDENT_ERR_P_HIGH,
          "identity forwards p error")
    _near(C(solve, "identity", axis=0, mask_param=2), IDENT_ERR_AXIS,
          "identity forwards axis error")


def check_empty_spec_shape(solve):
    _near(C(solve, "output", spec=[], variant="shared"), EMPTY_SPEC_SHARED,
          "empty spec shared")
    _near(C(solve, "output", spec=[]), EMPTY_SPEC_IID, "empty spec iid")


def check_empty_draws_shape(solve):
    _near(C(solve, "output", value_draws=[], start_draws=[]), EMPTY_DRAWS,
          "empty draw lists still have shape [0]")


def check_spec_string_rejected(solve):
    _raises(lambda: C(solve, "output", spec="abc"), "string spec",
            "rectangular")


def check_rand_lead23(solve):
    # n=6 default draws cross 1.0 in BOTH the value and start sequences.
    _near(C(solve, "rand_calls", spec=SPEC_LEAD23), RAND_LEAD23,
          "lead=(2,3) rand draws wrap at 1.0")


def check_mask_lead23(solve):
    _near(C(solve, "mask", spec=SPEC_LEAD23), MASK_LEAD23,
          "lead=(2,3) mask nesting with step 3")


def check_output_lead23(solve):
    _near(C(solve, "output", spec=SPEC_LEAD23), OUT_LEAD23,
          "lead=(2,3) output with asymmetric lead dims")


def check_param_negative_p(solve):
    # param mode bypasses the runner p guard: trunc(-1.2)=-1, floor(-1.2)=-2.
    _near(C(solve, "param", mask_param=10, p=-0.3, axis=2), PARAM_PNEG0_3,
          "negative p uses trunc, not floor")
    _near(C(solve, "param", mask_param=10, p=-0.7, axis=2), PARAM_PNEG0_7,
          "negative p second grid point")


def check_interval_freq_p05(solve):
    # axis=1 has length 3 while sh[-1] is 4; _get_mask_param must see 3.
    _near(C(solve, "interval", axis=1, p=0.5, mask_param=3), IV_FREQ_P05,
          "freq axis length feeds _get_mask_param")
    _near(C(solve, "mask", axis=1, p=0.5, mask_param=3), MASK_FREQ_P05,
          "freq axis p=0.5 mask")
    _near(C(solve, "output", axis=1, p=0.5, mask_param=3), OUT_FREQ_P05,
          "freq axis p=0.5 output")


def check_output_shared_freq_nonzero(solve):
    _near(C(solve, "output", variant="shared", axis=1, mask_value=-1.0,
            value_draws=[0.7], start_draws=[0.25]), OUT_SHARED_FREQ_NEG,
          "freq branch uses the caller's mask_value")


def check_output_int_spec_leaves(solve):
    _near(C(solve, "output", spec=[[1, 2, 3, 4], [5, 6, 7, 8],
                                   [9, 10, 11, 12]], variant="shared",
            value_draws=[0.7], start_draws=[0.25]),
          OUT_INT_SPEC, "integer spec is packed to floats")


def check_mask_value_int_is_float(solve):
    out = C(solve, "output", mask_value=7, value_draws=[0.7, 0.4],
            start_draws=[0.25, 0.6])
    _near(out, OUT_INT_VALUE, "integer fill values")

    def leaves(x):
        if isinstance(x, (list, tuple)):
            for v in x:
                for a in leaves(v):
                    yield a
        else:
            yield x

    for v in leaves(out):
        assert type(v) is float, \
            "output leaves must be Python floats, got %s" % type(v).__name__

