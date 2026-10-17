import itertools
import os
import sys

import numpy as np
import pytest

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf

from volterrasys._wavelet_ops import (
    analysis_axis,
    synthesis_axis,
    wavelet_to_natural_kernel,
)
from scripts.sparse_wavelet_selection import select_wavelet_coefficients
from volterrasys.Volterra1D import Volterra1D


def _known_kernel(order, kind):
    shape = (16,) + (16,) * order + (1,) * order + (1,)
    kernel = np.zeros(shape, dtype=np.float64)
    if kind == 'dense':
        kernel.fill(0.125)
    else:
        if kind == 'compressible':
            kernel.fill(1e-14)
        flat = kernel.reshape(-1)
        flat[[0, 17, 34]] = [1.0, -0.5, 0.25]
    return kernel


def _inputs():
    x = np.random.default_rng(2301).normal(size=(3, 16, 1))
    return x / np.linalg.norm(x, axis=(1, 2), keepdims=True) * 0.75


@pytest.mark.parametrize('order', [2, 3])
@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('kind', ['exactly_sparse', 'compressible', 'dense'])
def test_selection_meets_bound_and_measured_natural_error(
    order, wave, level, kind
):
    kernel = _known_kernel(order, kind)
    original = kernel.copy()
    tolerance = 0.0 if kind == 'dense' else 1e-8
    result = select_wavelet_coefficients(
        kernel,
        wave=wave,
        level=level,
        input_norm_bound=1.0,
        output_tolerance=tolerance,
        validation_inputs=_inputs(),
    )
    report = result.report
    assert report['bound_satisfied']
    assert report['status'] == 'bound_satisfied'
    assert report['validation_within_input_bound']
    assert report['validation_max_output_error'] <= tolerance + 1e-11
    assert report['validation_max_output_error'] <= (
        report['output_error_bound'] + 1e-11
    )
    expected_count = kernel.size if kind == 'dense' else 3
    assert report['retained_coefficients'] == expected_count
    assert report['retained_percentage'] == 100.0 * expected_count / kernel.size
    np.testing.assert_array_equal(kernel, original)
    rows = tuple(zip(*result.sparse_indices))
    np.testing.assert_array_equal(result.values, kernel[rows])
    assert not result.values.flags.writeable


def test_budget_failure_is_distinct_from_measured_accuracy():
    kernel = _known_kernel(2, 'dense')
    result = select_wavelet_coefficients(
        kernel,
        wave='bior2.2',
        level=2,
        input_norm_bound=1.0,
        output_tolerance=0.0,
        max_coefficients=5,
        validation_inputs=np.zeros((2, 16, 1)),
    )
    assert result.report['retained_coefficients'] == 5
    assert result.report['status'] == 'uncertified_at_budget'
    assert not result.report['bound_satisfied']
    assert result.report['validation_tolerance_met']
    assert result.report['validation_max_output_error'] == 0
    # Equal magnitudes use deterministic row-major ties.
    assert result.sparse_indices == tuple(
        tuple(int(v) for v in np.unravel_index(i, kernel.shape))
        for i in range(5)
    )


def test_zero_kernel_returns_a_compatible_zero_placeholder():
    kernel = np.zeros_like(_known_kernel(2, 'exactly_sparse'))
    result = select_wavelet_coefficients(
        kernel, wave='haar', input_norm_bound=1, output_tolerance=0
    )
    assert result.report['retained_coefficients'] == 1
    assert result.report['retained_nonzero_coefficients'] == 0
    assert result.report['output_error_bound'] == 0
    assert result.values[0] == 0


def test_validation_inputs_outside_radius_do_not_extend_the_bound():
    result = select_wavelet_coefficients(
        _known_kernel(2, 'dense'),
        wave='haar',
        input_norm_bound=0,
        output_tolerance=0,
        validation_inputs=_inputs(),
    )
    assert result.report['bound_satisfied']
    assert not result.report['validation_within_input_bound']
    assert not result.report['validation_tolerance_met']


def test_input_output_layout_matches_independent_multichannel_sums():
    order = 2
    kernel = np.zeros((4, 4, 4, 2, 2, 2), dtype=np.float64)
    kernel[0, 1, 2, 0, 1, 1] = 1.25
    kernel[1, 2, 0, 1, 0, 0] = -0.75
    kernel[3, 1, 3, 0, 0, 1] = 0.0001
    x = np.random.default_rng(2302).normal(size=(2, 4, 2)) * 0.1
    result = select_wavelet_coefficients(
        tf.constant(kernel), wave='haar', input_norm_bound=1,
        output_tolerance=1e-3, validation_inputs=x,
    )
    assert result.report['retained_coefficients'] == 2
    layer = Volterra1D(
        m=order, Ny=4, filters=2, wave='haar', backend='filterbank',
        kernel_parameterization='sparse_wavelet',
        sparse_indices=result.sparse_indices, dtype=tf.float64,
    )
    layer.build(x.shape)
    layer.sparse_kernel_values.assign(result.values)
    sparse_output = layer(x).numpy()
    natural = wavelet_to_natural_kernel(
        tf.constant(kernel), 'haar', order
    ).numpy()
    reference = np.zeros((2, 4, 2))
    for b, u, o, i, j, c, d in itertools.product(
        range(2), range(4), range(2), range(4), range(4), range(2), range(2)
    ):
        reference[b, u, o] += natural[u, i, j, c, d, o] * x[b, i, c] * x[b, j, d]
    errors = np.linalg.norm((reference - sparse_output).reshape((2, -1)), axis=1)
    np.testing.assert_allclose(
        result.report['validation_max_output_error'], max(errors), atol=1e-12
    )


def test_tail_error_is_not_lost_by_subtracting_large_energies():
    kernel = np.zeros((2, 2, 2, 1, 1, 1), dtype=np.float64)
    kernel.reshape(-1)[:2] = [1e150, 1.0]
    result = select_wavelet_coefficients(
        kernel, wave='haar', input_norm_bound=1,
        output_tolerance=0, max_coefficients=1,
    )
    assert result.report['discarded_coefficient_norm'] == 1.0
    assert not result.report['bound_satisfied']


def test_no_smaller_magnitude_prefix_satisfies_the_same_bound():
    kernel = _known_kernel(2, 'dense')
    kernel.reshape(-1)[:4] = [4, 3, 2, 1]
    result = select_wavelet_coefficients(
        kernel, wave='haar', input_norm_bound=1, output_tolerance=0.3
    )
    count = result.report['retained_coefficients']
    assert count > 1
    smaller = select_wavelet_coefficients(
        kernel, wave='haar', input_norm_bound=1, output_tolerance=0.3,
        max_coefficients=count - 1,
    )
    assert not smaller.report['bound_satisfied']


def test_reported_operator_bounds_dominate_spectral_norms():
    result = select_wavelet_coefficients(
        _known_kernel(2, 'dense'), wave='bior2.2', level=2,
        input_norm_bound=1, output_tolerance=0,
    )
    analysis = analysis_axis(tf.eye(16, dtype=tf.float64), 'bior2.2', 0, 2)
    synthesis = synthesis_axis(tf.eye(16, dtype=tf.float64), 'bior2.2', 0, 2)
    assert np.linalg.norm(analysis, 2) <= result.report['analysis_operator_norm_bound']
    assert np.linalg.norm(synthesis, 2) <= result.report['synthesis_operator_norm_bound']


@pytest.mark.parametrize('kwargs', [
    {'output_tolerance': -1},
    {'output_tolerance': float('nan')},
    {'input_norm_bound': float('inf')},
    {'max_coefficients': 0},
    {'max_coefficients': True},
    {'max_coefficients': 4097},
    {'wave': None},
    {'level': 0},
    {'validation_inputs': np.zeros((1, 16, 2))},
])
def test_invalid_selection_requests_are_rejected(kwargs):
    arguments = dict(wave='haar', input_norm_bound=1, output_tolerance=0.1)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        select_wavelet_coefficients(_known_kernel(2, 'dense'), **arguments)
