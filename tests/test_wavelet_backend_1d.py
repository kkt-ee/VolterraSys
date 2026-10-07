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

from volterrasys.LSIVolterra1D import LSIVolterra1D
from volterrasys.LinearVolterra1D import LinearVolterra1D
from volterrasys.QSIVolterra1D import QSIVolterra1D
from volterrasys.QuadraticVolterra1D import QuadraticVolterra1D
from volterrasys.Volterra1D import Volterra1D


CASES = (
    "lsi",
    "qsi",
    "linear",
    "quadratic",
    "volterra_m1",
    "volterra_m2",
    "volterra_m3",
)


def _make_layer(case, wave, backend):
    common = {"filters": 3, "wave": wave, "backend": backend}
    if case == "lsi":
        return LSIVolterra1D(kernel_size=3, **common)
    if case == "qsi":
        return QSIVolterra1D(kernel_size=3, **common)
    if case == "linear":
        return LinearVolterra1D(Ny=8, **common)
    if case == "quadratic":
        return QuadraticVolterra1D(Ny=8, **common)
    if case.startswith("volterra_m"):
        return Volterra1D(m=int(case[-1]), Ny=8, **common)
    raise AssertionError(f"Unknown test case: {case}")


def _kernel(layer):
    if isinstance(layer, (LSIVolterra1D, LinearVolterra1D)):
        return layer.h1
    if isinstance(layer, QSIVolterra1D):
        return layer.h
    if isinstance(layer, QuadraticVolterra1D):
        return layer.h2
    return layer.hm


def _transformed_kernel(layer):
    if isinstance(layer, LSIVolterra1D):
        return layer._LSIVolterra1D__makeH()
    if isinstance(layer, QSIVolterra1D):
        return layer.make_mra_h2()
    if isinstance(layer, LinearVolterra1D):
        return layer._LinearVolterra1D__make_H()
    if isinstance(layer, QuadraticVolterra1D):
        return layer._QuadraticVolterra1D__make_H()
    return layer._Volterra1D__make_H()


def _assert_close(actual, expected, tolerance=2e-5):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def _build_with_shared_kernel(case, wave):
    tf.random.set_seed(300)
    x = tf.random.normal((2, 8, 2))
    matrix = _make_layer(case, wave, "matrix")
    filterbank = _make_layer(case, wave, "filterbank")
    natural = _make_layer(case, None, "matrix")

    matrix(x)
    filterbank(x)
    natural(x)
    values = tf.random.normal(_kernel(matrix).shape) * 0.1
    _kernel(matrix).assign(values)
    _kernel(filterbank).assign(values)
    _kernel(natural).assign(values)
    return x, matrix, filterbank, natural


@pytest.mark.parametrize("wave", ["haar", "db2", "bior2.2", "rbio2.2"])
@pytest.mark.parametrize("case", CASES)
def test_filterbank_matches_matrix_and_natural_domain(case, wave):
    x, matrix, filterbank, natural = _build_with_shared_kernel(case, wave)

    _assert_close(_transformed_kernel(filterbank), _transformed_kernel(matrix))
    matrix_output = matrix(x)
    filterbank_output = filterbank(x)
    natural_output = natural(x)
    _assert_close(filterbank_output, matrix_output)
    _assert_close(filterbank_output, natural_output)

    assert filterbank.non_trainable_variables == []
    assert matrix.non_trainable_variables


@pytest.mark.parametrize("case", CASES)
def test_filterbank_gradients_match_matrix(case):
    x, matrix, filterbank, _ = _build_with_shared_kernel(case, "bior2.2")
    tf.random.set_seed(301)
    probe = tf.random.normal(matrix(x).shape)

    def gradients(layer):
        variable = tf.Variable(x)
        with tf.GradientTape() as tape:
            output = layer(variable)
            loss = tf.reduce_sum(output * probe)
        return tape.gradient(loss, [variable, _kernel(layer)])

    matrix_gradients = gradients(matrix)
    filterbank_gradients = gradients(filterbank)
    for filterbank_gradient, matrix_gradient in zip(
        filterbank_gradients, matrix_gradients
    ):
        _assert_close(filterbank_gradient, matrix_gradient, tolerance=3e-5)


@pytest.mark.parametrize(
    "layer_class, extra",
    [
        (LSIVolterra1D, {"kernel_size": 3}),
        (QSIVolterra1D, {"kernel_size": 3}),
        (LinearVolterra1D, {"Ny": 8}),
        (QuadraticVolterra1D, {"Ny": 8}),
        (Volterra1D, {"m": 3, "Ny": 8}),
    ],
)
def test_filterbank_backend_round_trips_through_config(layer_class, extra):
    layer = layer_class(
        filters=2,
        wave="bior2.2",
        backend="filterbank",
        **extra,
    )
    restored = layer_class.from_config(layer.get_config())
    assert restored.backend == "filterbank"


def test_filterbank_volterra_layer_runs_in_keras_graph():
    inputs = tf.keras.Input(shape=(8, 2))
    outputs = Volterra1D(
        m=3,
        filters=3,
        Ny=8,
        wave="bior2.2",
        backend="filterbank",
    )(inputs)
    model = tf.keras.Model(inputs, outputs)
    cloned_model = tf.keras.models.clone_model(model)

    tf.random.set_seed(302)
    x = tf.random.normal((2, 8, 2))
    assert model(x).shape == (2, 8, 3)
    assert cloned_model(x).shape == (2, 8, 3)


def test_filterbank_supports_a_different_output_length():
    tf.random.set_seed(303)
    x = tf.random.normal((1, 8, 1))
    matrix = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="matrix",
    )
    filterbank = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
    )
    natural = Volterra1D(m=3, filters=1, Ny=16, wave=None)

    matrix(x)
    filterbank(x)
    natural(x)
    values = tf.random.normal(matrix.hm.shape) * 0.1
    matrix.hm.assign(values)
    filterbank.hm.assign(values)
    natural.hm.assign(values)

    _assert_close(_transformed_kernel(filterbank), _transformed_kernel(matrix))
    _assert_close(filterbank(x), matrix(x))
    _assert_close(filterbank(x), natural(x))
    assert filterbank(x).shape == (1, 16, 1)


def test_backend_is_validated():
    with pytest.raises(ValueError, match="backend must be one of"):
        Volterra1D(backend="unknown")
