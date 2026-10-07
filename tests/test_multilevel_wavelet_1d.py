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
from TFDWT.multilevel.dwt import dwt, dwt_packed_axis, idwt_packed_axis

from volterrasys.LinearVolterra1D import LinearVolterra1D
from volterrasys.QuadraticVolterra1D import QuadraticVolterra1D
from volterrasys.Volterra1D import Volterra1D
from volterrasys._wavelet_ops import (
    natural_to_wavelet_kernel,
    wavelet_to_natural_kernel,
)


CASES = (
    ("linear", LinearVolterra1D, {}, 1),
    ("quadratic", QuadraticVolterra1D, {}, 2),
    ("volterra_m1", Volterra1D, {"m": 1}, 1),
    ("volterra_m2", Volterra1D, {"m": 2}, 2),
    ("volterra_m3", Volterra1D, {"m": 3}, 3),
)


def _kernel(layer):
    if isinstance(layer, LinearVolterra1D):
        return layer.h1
    if isinstance(layer, QuadraticVolterra1D):
        return layer.h2
    return layer.hm


def _assert_close(actual, expected, tolerance=4e-5):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
@pytest.mark.parametrize("name,layer_class,extra,order", CASES)
def test_multilevel_direct_H_matches_matrix_filterbank_and_natural(
    name,
    layer_class,
    extra,
    order,
    wave,
    level,
):
    del name
    tf.keras.utils.set_random_seed(410 + level + order)
    x = tf.random.normal((2, 32, 1))
    common = {
        "filters": 1,
        "Ny": 32,
        "wave": wave,
        "level": level,
    }
    matrix = layer_class(backend="matrix", **common, **extra)
    filterbank = layer_class(backend="filterbank", **common, **extra)
    natural = layer_class(
        filters=1,
        Ny=32,
        wave=None,
        level=level,
        **extra,
    )

    matrix(x)
    filterbank(x)
    natural(x)
    h = tf.random.normal(_kernel(natural).shape) * 0.01
    H = natural_to_wavelet_kernel(h, wave, order, level=level)
    _kernel(matrix).assign(H)
    _kernel(filterbank).assign(H)
    _kernel(natural).assign(h)

    _assert_close(_kernel(matrix), H, tolerance=0.0)
    _assert_close(_kernel(filterbank), H, tolerance=0.0)
    _assert_close(
        wavelet_to_natural_kernel(H, wave, order, level=level),
        h,
    )
    _assert_close(matrix(x), filterbank(x))
    _assert_close(matrix(x), natural(x))
    _assert_close(filterbank(x), natural(x))


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
@pytest.mark.parametrize("level", [2, 3])
def test_packed_coefficients_match_each_multilevel_subband(backend, level):
    tf.keras.utils.set_random_seed(420 + level)
    x = tf.random.normal((2, 32, 2))
    subbands = dwt(x, level=level, Ψ="bior2.2", backend=backend)
    expected = tf.concat(
        [subbands[-1]] + list(reversed(subbands[:-1])),
        axis=1,
    )
    packed = dwt_packed_axis(
        x,
        level=level,
        wave="bior2.2",
        axis=1,
        backend=backend,
    )

    _assert_close(packed, expected)
    _assert_close(
        idwt_packed_axis(
            packed,
            level=level,
            wave="bior2.2",
            axis=1,
            backend=backend,
        ),
        x,
    )


def test_multilevel_filterbank_gradients_match_matrix():
    tf.keras.utils.set_random_seed(430)
    x = tf.random.normal((2, 16, 1))
    probe = tf.random.normal((2, 16, 1))
    matrix = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="matrix",
        level=2,
    )
    filterbank = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
    )
    matrix(x)
    filterbank(x)
    H = tf.random.normal(matrix.hm.shape) * 0.01
    matrix.hm.assign(H)
    filterbank.hm.assign(H)

    def gradients(layer):
        variable = tf.Variable(x)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(variable) * probe)
        return tape.gradient(loss, [variable, layer.hm])

    for matrix_gradient, filterbank_gradient in zip(
        gradients(matrix),
        gradients(filterbank),
    ):
        _assert_close(matrix_gradient, filterbank_gradient)


def test_multilevel_supports_different_input_and_output_lengths():
    tf.keras.utils.set_random_seed(440)
    x = tf.random.normal((1, 32, 1))
    matrix = Volterra1D(
        m=2,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="matrix",
        level=2,
    )
    filterbank = Volterra1D(
        m=2,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
    )
    natural = Volterra1D(m=2, filters=1, Ny=16, wave=None, level=2)
    matrix(x)
    filterbank(x)
    natural(x)
    h = tf.random.normal(natural.hm.shape) * 0.01
    H = natural_to_wavelet_kernel(h, "bior2.2", 2, level=2)
    matrix.hm.assign(H)
    filterbank.hm.assign(H)
    natural.hm.assign(h)

    assert matrix(x).shape == (1, 16, 1)
    _assert_close(matrix(x), filterbank(x))
    _assert_close(matrix(x), natural(x))


@pytest.mark.parametrize(
    "layer",
    [
        LinearVolterra1D(level=3),
        QuadraticVolterra1D(level=3),
        Volterra1D(m=3, level=3),
    ],
)
def test_level_round_trips_through_config(layer):
    restored = type(layer).from_config(layer.get_config())
    assert restored.level == 3


def test_multilevel_model_round_trips_through_keras_save(tmp_path):
    inputs = tf.keras.Input(shape=(16, 1))
    outputs = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
    )(inputs)
    model = tf.keras.Model(inputs, outputs)
    x = tf.random.normal((1, 16, 1))
    expected = model(x)
    path = tmp_path / "multilevel.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)

    _assert_close(restored(x), expected, tolerance=0.0)
    restored_layer = next(
        layer for layer in restored.layers if isinstance(layer, Volterra1D)
    )
    assert restored_layer.level == 2


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
@pytest.mark.parametrize(
    "layer_class,extra",
    [
        (LinearVolterra1D, {}),
        (QuadraticVolterra1D, {}),
        (Volterra1D, {"m": 2}),
    ],
)
def test_level_is_validated(layer_class, extra, invalid):
    with pytest.raises(ValueError, match="level must be a positive integer"):
        layer_class(level=invalid, **extra)


def test_existing_positional_backend_argument_is_preserved():
    assert LinearVolterra1D(1, 8, "haar", "filterbank").backend == "filterbank"
    assert QuadraticVolterra1D(1, 8, "haar", "filterbank").backend == "filterbank"
    assert Volterra1D(2, 1, 8, "haar", "filterbank").backend == "filterbank"

