import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from volterrasys.Volterra1D import Volterra1D
from volterrasys._wavelet_ops import (
    analysis_axis,
    synthesis_axis,
)


def _assert_close(actual, expected, tolerance=4e-5):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def _dense_reference(layer, x):
    coordinates = x
    if layer.mra:
        coordinates = analysis_axis(
            x,
            layer.wave,
            axis=1,
            level=layer.level,
        )
    kernel = layer.materialize_kernel()
    output = tf.einsum(
        layer._contraction_equation,
        *([coordinates] * layer.m),
        kernel,
    )
    if layer.mra:
        output = synthesis_axis(
            output,
            layer.wave,
            axis=1,
            level=layer.level,
        )
    return output


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize(
    "wave,level",
    [(None, 1), ("bior2.2", 1), ("bior2.2", 2)],
)
def test_factorized_values_and_gradients_match_materialized_kernel(
    m,
    wave,
    level,
):
    tf.keras.utils.set_random_seed(510 + m + level)
    x_value = tf.random.normal((2, 16, 2)) * 0.2
    layer = Volterra1D(
        m=m,
        filters=2,
        Ny=16,
        wave=wave,
        backend="filterbank",
        level=level,
        kernel_parameterization="symmetric_cp",
        kernel_rank=3,
    )
    layer(x_value)
    probe = tf.random.normal((2, 16, 2))

    x_factorized = tf.Variable(x_value)
    with tf.GradientTape() as tape:
        factorized_loss = tf.reduce_sum(layer(x_factorized) * probe)
    factorized_gradients = tape.gradient(
        factorized_loss,
        [x_factorized, layer.input_factors, layer.output_factors],
    )

    x_dense = tf.Variable(x_value)
    with tf.GradientTape() as tape:
        dense_output = _dense_reference(layer, x_dense)
        dense_loss = tf.reduce_sum(dense_output * probe)
    dense_gradients = tape.gradient(
        dense_loss,
        [x_dense, layer.input_factors, layer.output_factors],
    )

    _assert_close(layer(x_value), dense_output)
    for factorized_gradient, dense_gradient in zip(
        factorized_gradients,
        dense_gradients,
    ):
        _assert_close(factorized_gradient, dense_gradient)


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
@pytest.mark.parametrize("level", [1, 2])
def test_factorized_wavelet_and_natural_coordinates_are_equivalent(
    backend,
    level,
):
    tf.keras.utils.set_random_seed(520 + level)
    x = tf.random.normal((2, 16, 2)) * 0.2
    layer = Volterra1D(
        m=3,
        filters=2,
        Ny=16,
        wave="bior2.2",
        backend=backend,
        level=level,
        kernel_parameterization="symmetric_cp",
        kernel_rank=2,
    )
    wavelet_output = layer(x)
    checked_wavelet_output, natural_output = layer.sanity_check()

    _assert_close(checked_wavelet_output, wavelet_output)
    _assert_close(wavelet_output, natural_output)
    natural_kernel = layer.materialize_kernel(domain="natural")
    dense_natural_output = tf.einsum(
        layer._contraction_equation,
        *([x] * layer.m),
        natural_kernel,
    )
    _assert_close(wavelet_output, dense_natural_output)


def test_factorized_parameter_count_uses_rank_not_tensor_order():
    x = tf.zeros((1, 3, 2))
    layer = Volterra1D(
        m=4,
        filters=2,
        Ny=4,
        wave=None,
        kernel_parameterization="symmetric_cp",
        kernel_rank=3,
    )
    layer(x)

    input_size = 3 * 2
    output_size = 4 * 2
    assert sum(np.prod(variable.shape) for variable in layer.trainable_weights) == (
        3 * (input_size + output_size)
    )
    assert layer.materialize_kernel().shape == (4, 3, 3, 3, 3, 2, 2, 2, 2, 2)


def test_factorized_configuration_and_model_round_trip(tmp_path):
    inputs = tf.keras.Input(shape=(16, 2))
    outputs = Volterra1D(
        m=3,
        filters=2,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
        kernel_parameterization="symmetric_cp",
        kernel_rank=3,
    )(inputs)
    model = tf.keras.Model(inputs, outputs)
    x = tf.random.normal((1, 16, 2))
    expected = model(x)
    path = tmp_path / "factorized.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    restored_layer = next(
        candidate
        for candidate in restored.layers
        if isinstance(candidate, Volterra1D)
    )

    _assert_close(restored(x), expected, tolerance=0.0)
    assert restored_layer.kernel_parameterization == "symmetric_cp"
    assert restored_layer.kernel_rank == 3


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"kernel_parameterization": "unknown"},
            "kernel_parameterization must be",
        ),
        (
            {"kernel_parameterization": "symmetric_cp"},
            "kernel_rank must be a positive integer",
        ),
        (
            {
                "kernel_parameterization": "symmetric_cp",
                "kernel_rank": 0,
            },
            "kernel_rank must be a positive integer",
        ),
        (
            {"kernel_rank": 2},
            "kernel_rank is only used",
        ),
    ],
)
def test_factorized_arguments_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Volterra1D(**kwargs)
