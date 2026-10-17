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


def _indices(order):
    return [
        [0] + [0] * order + [0] * order + [0],
        [0] + list(range(1, order + 1)) + [1] * order + [0],
        [3] + list(range(2, order + 2)) + [0] * order + [1],
        [5] + list(range(3, order + 3)) + [j % 2 for j in range(order)] + [1],
    ]


def _assert_close(actual, expected, tolerance=5e-5):
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def _materialize(indices, values, shape):
    kernel = np.zeros(tuple(shape), dtype=np.float32)
    for index, value in zip(indices, values.numpy()):
        kernel[tuple(index)] = value
    return tf.constant(kernel)


def _make_sparse_and_dense(
    order, backend="filterbank", level=2, wave="bior2.2"
):
    common = {
        "m": order,
        "filters": 2,
        "Ny": 16,
        "wave": wave,
        "backend": backend,
        "level": level,
    }
    sparse = Volterra1D(
        **common,
        kernel_parameterization="sparse_wavelet",
        sparse_indices=_indices(order),
    )
    dense = Volterra1D(**common)
    return sparse, dense


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
@pytest.mark.parametrize("wave", ["haar", "bior2.2"])
def test_sparse_wavelet_matches_its_dense_kernel(order, backend, wave):
    tf.keras.utils.set_random_seed(610 + order)
    x = tf.random.normal((2, 16, 2)) * 0.2
    sparse, dense = _make_sparse_and_dense(order, backend=backend, wave=wave)
    sparse(x)
    dense(x)

    values = tf.constant([0.75, -0.5, 0.25, 1.25])
    sparse.sparse_kernel_values.assign(values)
    dense_kernel = _materialize(_indices(order), values, dense.hm.shape)
    dense.hm.assign(dense_kernel)

    _assert_close(sparse.materialize_kernel(), dense_kernel, tolerance=0.0)
    _assert_close(sparse(x), dense(x))


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
def test_sparse_wavelet_values_and_input_gradients_match_dense(backend):
    tf.keras.utils.set_random_seed(620)
    x_value = tf.random.normal((2, 16, 2)) * 0.2
    probe = tf.random.normal((2, 16, 2))
    sparse, dense = _make_sparse_and_dense(2, backend=backend)
    sparse(x_value)
    dense(x_value)

    values = tf.constant([0.75, -0.5, 0.25, 1.25])
    sparse.sparse_kernel_values.assign(values)
    dense.hm.assign(_materialize(_indices(2), values, dense.hm.shape))

    sparse_input = tf.Variable(x_value)
    with tf.GradientTape() as tape:
        sparse_loss = tf.reduce_sum(sparse(sparse_input) * probe)
    sparse_input_gradient, sparse_value_gradient = tape.gradient(
        sparse_loss,
        [sparse_input, sparse.sparse_kernel_values],
    )

    dense_input = tf.Variable(x_value)
    with tf.GradientTape() as tape:
        dense_loss = tf.reduce_sum(dense(dense_input) * probe)
    dense_input_gradient, dense_kernel_gradient = tape.gradient(
        dense_loss,
        [dense_input, dense.hm],
    )

    _assert_close(sparse_input_gradient, dense_input_gradient)
    _assert_close(
        sparse_value_gradient,
        tf.gather_nd(dense_kernel_gradient, tf.constant(_indices(2))),
    )


@pytest.mark.parametrize("level", [1, 2])
def test_sparse_wavelet_and_materialized_natural_kernel_are_equivalent(level):
    tf.keras.utils.set_random_seed(630 + level)
    x = tf.random.normal((2, 16, 2)) * 0.2
    sparse, _ = _make_sparse_and_dense(2, level=level)
    sparse(x)
    sparse.sparse_kernel_values.assign([0.75, -0.5, 0.25, 1.25])

    wavelet_output = sparse(x)
    natural_kernel = sparse.materialize_kernel(domain="natural")
    natural_output = tf.einsum(
        sparse._contraction_equation,
        x,
        x,
        natural_kernel,
    )
    _assert_close(wavelet_output, natural_output)


def test_sparse_wavelet_stores_only_active_trainable_values():
    x = tf.zeros((1, 16, 2))
    sparse, _ = _make_sparse_and_dense(3)
    sparse(x)

    assert sparse.sparse_kernel_values.shape == (4,)
    assert sum(
        np.prod(variable.shape) for variable in sparse.trainable_weights
    ) == 4
    assert sparse.materialize_kernel().shape == (
        16,
        16,
        16,
        16,
        2,
        2,
        2,
        2,
    )


def test_sparse_wavelet_runs_when_the_dense_size_exceeds_int64():
    order = 16
    layer = Volterra1D(
        m=order,
        filters=1,
        Ny=8,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_indices=[[0] + [0] * order + [0] * order + [0]],
    )
    # The dense tensor would contain 8 * (16 * 2)**16 entries.
    x = tf.ones((1, 16, 2)) * 0.1
    output = layer(x)
    assert output.shape == (1, 8, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(output)))
    assert layer.sparse_kernel_values.shape == (1,)
    with pytest.raises(ValueError, match="too large to materialize"):
        layer.materialize_kernel()


def test_sparse_wavelet_traces_with_dynamic_batch_and_shared_output_indices():
    layer, _ = _make_sparse_and_dense(2)

    @tf.function(input_signature=[tf.TensorSpec((None, 16, 2), tf.float32)])
    def evaluate(x):
        return layer(x)

    for batch_size in (1, 3):
        x = tf.random.normal((batch_size, 16, 2)) * 0.2
        _assert_close(evaluate(x), layer(x))

    operation_types = {
        operation.type
        for operation in evaluate.get_concrete_function().graph.get_operations()
    }
    assert "UnsortedSegmentSum" in operation_types
    assert "ScatterNd" not in operation_types


def test_sparse_wavelet_configuration_and_model_round_trip(tmp_path):
    indices = _indices(2)
    inputs = tf.keras.Input(shape=(16, 2))
    layer = Volterra1D(
        m=2,
        filters=2,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
        kernel_parameterization="sparse_wavelet",
        sparse_indices=indices,
    )
    model = tf.keras.Model(inputs, layer(inputs))
    layer.sparse_kernel_values.assign([0.75, -0.5, 0.25, 1.25])
    x = tf.random.normal((1, 16, 2))
    expected = model(x)

    path = tmp_path / "sparse_wavelet.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    restored_layer = next(
        candidate
        for candidate in restored.layers
        if isinstance(candidate, Volterra1D)
    )

    _assert_close(restored(x), expected, tolerance=0.0)
    assert restored_layer.kernel_parameterization == "sparse_wavelet"
    assert restored_layer.sparse_indices == tuple(
        tuple(row) for row in indices
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "wave": None,
                "kernel_parameterization": "sparse_wavelet",
                "sparse_indices": [[0, 0, 0, 0]],
            },
            "requires a wavelet name",
        ),
        (
            {"kernel_parameterization": "sparse_wavelet"},
            "sparse_indices must be provided",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_indices": [[0, 0, 0]],
            },
            "must contain 4 entries",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_indices": [[0, 0, 0, 0], [0, 0, 0, 0]],
            },
            "must be unique",
        ),
        (
            {"sparse_indices": [[0, 0, 0, 0]]},
            "sparse_indices is only used",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "kernel_rank": 2,
                "sparse_indices": [[0, 0, 0, 0]],
            },
            "kernel_rank is only used",
        ),
    ],
)
def test_sparse_wavelet_arguments_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Volterra1D(**kwargs)


def test_sparse_wavelet_indices_are_checked_against_built_shape():
    layer = Volterra1D(
        m=1,
        filters=1,
        Ny=4,
        wave="haar",
        kernel_parameterization="sparse_wavelet",
        sparse_indices=[[0, 4, 0, 0]],
    )
    with pytest.raises(ValueError, match="outside axis 1 with size 4"):
        layer(tf.zeros((1, 4, 1)))
