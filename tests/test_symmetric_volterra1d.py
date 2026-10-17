import math
import os
import sys
from itertools import permutations

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tensorflow as tf

from volterrasys.Volterra1D import Volterra1D
from volterrasys._wavelet_ops import natural_to_wavelet_kernel


def _assert_close(actual, expected, tolerance=6e-5):
    actual = tf.convert_to_tensor(actual)
    expected = tf.convert_to_tensor(expected)
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def _dense_reference(layer, coordinates):
    return tf.einsum(
        layer._contraction_equation,
        *([coordinates] * layer.m),
        layer.materialize_kernel(
            domain="wavelet" if layer.mra else "natural"
        ),
    )


def _canonical_values(layer, dense_kernel):
    values = np.empty(
        (
            layer.Ny,
            len(layer._symmetric_input_tuples),
            layer.filters,
        ),
        dtype=np.float32,
    )
    dense_kernel = np.asarray(dense_kernel)
    for output_index in range(layer.Ny):
        for tuple_index, indices in enumerate(
            layer._symmetric_input_tuples
        ):
            coordinates = tuple(
                divmod(index, layer.channels) for index in indices
            )
            positions = tuple(position for position, _ in coordinates)
            channels = tuple(channel for _, channel in coordinates)
            for filter_index in range(layer.filters):
                values[output_index, tuple_index, filter_index] = (
                    dense_kernel[
                        (output_index,)
                        + positions
                        + channels
                        + (filter_index,)
                    ]
                )
    return values


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(
    "wave,backend,level",
    [
        (None, "matrix", 1),
        ("haar", "matrix", 1),
        ("bior2.2", "filterbank", 1),
        ("bior2.2", "filterbank", 2),
    ],
)
def test_symmetric_forward_and_gradients_match_dense_reference(
    order,
    wave,
    backend,
    level,
):
    rng = np.random.default_rng(1100 + order)
    length = 16 if level == 2 else 8
    x = tf.Variable(
        rng.normal(size=(2, length, 2)).astype(np.float64)
    )
    layer = Volterra1D(
        m=order,
        filters=2,
        Ny=length,
        wave=wave,
        backend=backend,
        level=level,
        kernel_parameterization="symmetric",
        dtype="float64",
    )
    layer(x)
    layer.symmetric_kernel_values.assign(
        rng.normal(size=layer.symmetric_kernel_values.shape).astype(
            np.float64
        )
    )

    coordinates = tf.Variable(
        rng.normal(size=(2, length, 2)).astype(np.float64)
    )

    with tf.GradientTape() as direct_tape:
        direct = layer._contract_symmetric(coordinates)
        direct_loss = tf.reduce_sum(direct ** 2)
    direct_gradients = direct_tape.gradient(
        direct_loss,
        [coordinates, layer.symmetric_kernel_values],
    )

    with tf.GradientTape() as dense_tape:
        dense = _dense_reference(layer, coordinates)
        dense_loss = tf.reduce_sum(dense ** 2)
    dense_gradients = dense_tape.gradient(
        dense_loss,
        [coordinates, layer.symmetric_kernel_values],
    )

    _assert_close(direct, dense)
    for direct_gradient, dense_gradient in zip(
        direct_gradients,
        dense_gradients,
    ):
        _assert_close(direct_gradient, dense_gradient, tolerance=1e-9)


def test_symmetric_materialization_has_identical_permutations():
    x = tf.zeros((1, 3, 2), dtype=tf.float32)
    layer = Volterra1D(
        m=3,
        filters=1,
        Ny=2,
        wave=None,
        kernel_parameterization="symmetric",
    )
    layer(x)
    values = np.arange(
        np.prod(layer.symmetric_kernel_values.shape),
        dtype=np.float32,
    ).reshape(layer.symmetric_kernel_values.shape)
    layer.symmetric_kernel_values.assign(values)
    kernel = layer.materialize_kernel().numpy()

    for output_index in range(layer.Ny):
        for tuple_index, indices in enumerate(
            layer._symmetric_input_tuples
        ):
            coordinates = tuple(
                divmod(index, layer.channels) for index in indices
            )
            expected = values[output_index, tuple_index, 0]
            for permutation in set(permutations(coordinates)):
                positions = tuple(position for position, _ in permutation)
                channels = tuple(channel for _, channel in permutation)
                assert kernel[
                    (output_index,) + positions + channels + (0,)
                ] == expected


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
@pytest.mark.parametrize(
    "wave,level",
    [("haar", 1), ("bior2.2", 1), ("bior2.2", 2)],
)
def test_symmetric_natural_and_wavelet_outputs_match(
    backend,
    wave,
    level,
):
    rng = np.random.default_rng(1200 + level)
    length = 16 if level == 2 else 8
    x = tf.constant(
        rng.normal(size=(2, length, 2)).astype(np.float32)
    )
    common = dict(
        m=3,
        filters=2,
        Ny=length,
        kernel_parameterization="symmetric",
    )
    natural = Volterra1D(wave=None, **common)
    wavelet = Volterra1D(
        wave=wave,
        backend=backend,
        level=level,
        **common,
    )
    natural(x)
    wavelet(x)

    natural_values = rng.normal(
        size=natural.symmetric_kernel_values.shape
    ).astype(np.float32)
    natural.symmetric_kernel_values.assign(natural_values)
    h = natural.materialize_kernel(domain="natural")
    H = natural_to_wavelet_kernel(
        h,
        wave,
        order=3,
        level=level,
    )
    wavelet.symmetric_kernel_values.assign(_canonical_values(wavelet, H))

    _assert_close(natural(x), wavelet(x), tolerance=2e-4)
    transformed_output, natural_output = wavelet.sanity_check()
    _assert_close(transformed_output, natural_output, tolerance=2e-4)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_symmetric_parameter_count_is_exact(order):
    length = 4
    channels = 2
    outputs = 3
    filters = 2
    x = tf.zeros((1, length, channels), dtype=tf.float32)
    layer = Volterra1D(
        m=order,
        filters=filters,
        Ny=outputs,
        wave=None,
        kernel_parameterization="symmetric",
    )
    layer(x)
    expected = (
        outputs
        * filters
        * math.comb(length * channels + order - 1, order)
    )
    assert layer.count_params() == expected
    assert layer.symmetric_kernel_values.shape == (
        outputs,
        math.comb(length * channels + order - 1, order),
        filters,
    )


@pytest.mark.parametrize("wave", [None, "bior2.2"])
def test_symmetric_configuration_and_model_round_trip(tmp_path, wave):
    length = 16 if wave else 8
    inputs = tf.keras.Input(shape=(length, 2))
    layer = Volterra1D(
        m=3,
        filters=2,
        Ny=length,
        wave=wave,
        backend="filterbank" if wave else "matrix",
        level=2,
        kernel_parameterization="symmetric",
    )
    model = tf.keras.Model(inputs, layer(inputs))
    x = tf.random.stateless_normal((2, length, 2), seed=(17, 19))
    expected = model(x)
    path = tmp_path / "symmetric.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    restored_layer = next(
        candidate
        for candidate in restored.layers
        if isinstance(candidate, Volterra1D)
    )

    _assert_close(restored(x), expected, tolerance=0.0)
    assert restored_layer.kernel_parameterization == "symmetric"
    assert restored_layer.sparse_indices is None


def test_symmetric_traces_without_dense_materialization():
    layer = Volterra1D(
        m=3,
        filters=2,
        Ny=4,
        wave=None,
        kernel_parameterization="symmetric",
    )

    @tf.function(input_signature=[tf.TensorSpec((None, 4, 2), tf.float32)])
    def evaluate(x):
        return layer(x)

    concrete = evaluate.get_concrete_function()
    operation_types = {
        operation.type for operation in concrete.graph.get_operations()
    }
    assert "ScatterNd" not in operation_types
    assert evaluate(tf.ones((3, 4, 2))).shape == (3, 4, 2)


def test_symmetric_rejects_unrelated_arguments():
    with pytest.raises(ValueError, match="kernel_rank is only used"):
        Volterra1D(
            kernel_parameterization="symmetric",
            kernel_rank=2,
        )
    with pytest.raises(ValueError, match="sparse_indices is only used"):
        Volterra1D(
            kernel_parameterization="symmetric",
            sparse_indices=[[0, 0, 0, 0]],
        )
