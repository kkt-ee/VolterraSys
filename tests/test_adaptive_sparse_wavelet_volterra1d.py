import math
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

from volterrasys.Volterra1D import (
    Volterra1D,
    _rank_canonical_tuple,
    _unrank_canonical_tuple,
)
from volterrasys.adaptive_wavelet_support import AdaptiveWaveletSupport


def _assert_close(actual, expected, tolerance=2e-5):
    np.testing.assert_allclose(
        tf.convert_to_tensor(actual).numpy(),
        tf.convert_to_tensor(expected).numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def _dense_reference(layer, coordinates):
    return tf.einsum(
        layer._contraction_equation,
        *([coordinates] * layer.m),
        layer.materialize_kernel(domain="wavelet"),
    )


def test_canonical_tuple_rank_and_unrank_are_exact_inverses():
    for input_size in range(1, 7):
        for order in range(1, 5):
            tuple_count = math.comb(input_size + order - 1, order)
            for rank in range(tuple_count):
                row = _unrank_canonical_tuple(rank, input_size, order)
                assert _rank_canonical_tuple(row, input_size) == rank


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize(
    "wave,backend,level",
    [
        ("haar", "matrix", 1),
        ("bior2.2", "filterbank", 1),
        ("bior2.2", "filterbank", 2),
    ],
)
def test_adaptive_sparse_contraction_and_gradients_match_dense(
    order,
    wave,
    backend,
    level,
):
    rng = np.random.default_rng(2100 + order + level)
    length = 16
    layer = Volterra1D(
        m=order,
        filters=2,
        Ny=length,
        wave=wave,
        backend=backend,
        level=level,
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=23,
        sparse_seed=17,
        dtype="float64",
    )
    layer(tf.zeros((1, length, 2), dtype=tf.float64))
    layer.sparse_kernel_values.assign(
        rng.normal(size=(23,)).astype(np.float64)
    )
    coordinates = tf.Variable(
        rng.normal(size=(3, length, 2)).astype(np.float64)
    )

    with tf.GradientTape() as direct_tape:
        direct = layer._contract_adaptive_sparse(coordinates)
        direct_loss = tf.reduce_sum(direct ** 2)
    direct_gradients = direct_tape.gradient(
        direct_loss,
        [coordinates, layer.sparse_kernel_values],
    )

    with tf.GradientTape() as dense_tape:
        dense = _dense_reference(layer, coordinates)
        dense_loss = tf.reduce_sum(dense ** 2)
    dense_gradients = dense_tape.gradient(
        dense_loss,
        [coordinates, layer.sparse_kernel_values],
    )

    _assert_close(direct, dense, tolerance=1e-10)
    for direct_gradient, dense_gradient in zip(
        direct_gradients,
        dense_gradients,
    ):
        _assert_close(direct_gradient, dense_gradient, tolerance=1e-9)


@pytest.mark.parametrize("backend", ["matrix", "filterbank"])
@pytest.mark.parametrize("wave,level", [("haar", 1), ("bior2.2", 2)])
def test_adaptive_sparse_wavelet_output_matches_dense_layer(
    backend,
    wave,
    level,
):
    rng = np.random.default_rng(2200 + level)
    x = tf.Variable(rng.normal(size=(2, 16, 2)).astype(np.float64))
    adaptive = Volterra1D(
        m=3,
        filters=2,
        Ny=16,
        wave=wave,
        backend=backend,
        level=level,
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=29,
        sparse_seed=23,
        dtype="float64",
    )
    dense = Volterra1D(
        m=3,
        filters=2,
        Ny=16,
        wave=wave,
        backend=backend,
        level=level,
        kernel_parameterization="dense",
        dtype="float64",
    )
    adaptive(x)
    dense(x)
    adaptive.sparse_kernel_values.assign(
        rng.normal(size=(29,)).astype(np.float64)
    )
    dense.hm.assign(adaptive.materialize_kernel(domain="wavelet"))

    with tf.GradientTape() as adaptive_tape:
        adaptive_output = adaptive(x)
        adaptive_loss = tf.reduce_sum(adaptive_output ** 2)
    adaptive_input_gradient = adaptive_tape.gradient(adaptive_loss, x)

    with tf.GradientTape() as dense_tape:
        dense_output = dense(x)
        dense_loss = tf.reduce_sum(dense_output ** 2)
    dense_input_gradient = dense_tape.gradient(dense_loss, x)

    _assert_close(adaptive_output, dense_output, tolerance=2e-8)
    _assert_close(
        adaptive_input_gradient,
        dense_input_gradient,
        tolerance=2e-8,
    )


def test_adaptive_support_is_canonical_unique_and_fixed_size():
    length = 8
    channels = 2
    outputs = 6
    filters = 3
    budget = 31
    layer = Volterra1D(
        m=3,
        filters=filters,
        Ny=outputs,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=budget,
        sparse_seed=29,
    )
    layer(tf.zeros((1, length, channels)))
    support = layer.adaptive_sparse_support()

    assert len(support) == budget
    assert len(set(support)) == budget
    assert all(tuple(sorted(row[1:])) == row[1:] for row in support)
    assert all(0 <= row[0] < outputs * filters for row in support)
    assert all(
        0 <= index < length * channels
        for row in support
        for index in row[1:]
    )
    assert sum(
        math.prod(weight.shape) for weight in layer.trainable_weights
    ) == budget
    assert layer.adaptive_sparse_indices.shape == (budget, 4)
    assert layer.adaptive_sparse_multiplicities.shape == (budget,)


def test_adaptive_support_replacement_does_not_retrace():
    layer = Volterra1D(
        m=2,
        filters=1,
        Ny=8,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=7,
        sparse_seed=31,
    )

    @tf.function(input_signature=[tf.TensorSpec((None, 8, 2), tf.float32)])
    def evaluate(x):
        return layer(x)

    x = tf.random.stateless_normal((2, 8, 2), seed=(3, 5))
    evaluate(x)
    layer.sparse_kernel_values.assign(tf.ones((7,)))
    before = evaluate(x)
    tracing_count = evaluate.experimental_get_tracing_count()
    replacement = layer.sample_inactive_sparse_support(1, seed=37)
    layer.replace_adaptive_sparse_support((0,), replacement)
    after = evaluate(x)

    assert evaluate.experimental_get_tracing_count() == tracing_count
    assert layer.adaptive_sparse_support()[0] == replacement[0]
    assert layer.sparse_kernel_values[0] == 0
    assert not np.allclose(before.numpy(), after.numpy())


def test_adaptive_sparse_model_and_support_round_trip(tmp_path):
    inputs = tf.keras.Input(shape=(16, 2))
    layer = Volterra1D(
        m=3,
        filters=2,
        Ny=16,
        wave="bior2.2",
        backend="filterbank",
        level=2,
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=17,
        sparse_seed=41,
    )
    model = tf.keras.Model(inputs, layer(inputs))
    x = tf.random.stateless_normal((2, 16, 2), seed=(7, 11))
    replacement = layer.sample_inactive_sparse_support(2, seed=43)
    layer.replace_adaptive_sparse_support(
        (0, 1),
        replacement,
        values=(1.25, -0.75),
        increment_update_count=True,
    )
    expected = model(x)
    expected_support = layer.adaptive_sparse_support()
    path = tmp_path / "adaptive_sparse.keras"
    model.save(path)
    restored = tf.keras.models.load_model(path)
    restored_layer = next(
        candidate
        for candidate in restored.layers
        if isinstance(candidate, Volterra1D)
    )

    _assert_close(restored(x), expected, tolerance=0.0)
    assert restored_layer.sparse_support == "adaptive"
    assert restored_layer.sparse_budget == 17
    assert restored_layer.adaptive_sparse_support() == expected_support
    assert int(restored_layer.adaptive_sparse_update_count.numpy()) == 1
    assert (
        restored_layer.sample_inactive_sparse_support(3)
        == layer.sample_inactive_sparse_support(3)
    )


def test_adaptive_support_builds_without_enumerating_full_kernel():
    layer = Volterra1D(
        m=10,
        filters=1,
        Ny=16,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=3,
        sparse_seed=47,
    )
    output = layer(tf.ones((1, 16, 1)))

    assert output.shape == (1, 16, 1)
    assert layer.sparse_kernel_values.shape == (3,)
    assert layer._adaptive_support_size == 16 * math.comb(25, 10)


def test_adaptive_controller_grows_gradient_selected_support_and_resets_adam():
    x = tf.constant(
        [[[1.0], [0.0]], [[2.0], [0.0]], [[-1.0], [0.0]]]
    )
    layer = Volterra1D(
        m=1,
        filters=1,
        Ny=2,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=2,
        sparse_seed=0,
    )
    model = tf.keras.Sequential([tf.keras.Input((2, 1)), layer])
    model(x)
    layer.replace_adaptive_sparse_support(
        (0, 1),
        ((0, 0), (0, 1)),
        values=(0.0, 0.0),
    )

    target_layer = Volterra1D(
        m=1,
        filters=1,
        Ny=2,
        wave="haar",
        backend="filterbank",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=1,
        sparse_seed=1,
    )
    target_model = tf.keras.Sequential(
        [tf.keras.Input((2, 1)), target_layer]
    )
    target_model(x)
    target_layer.replace_adaptive_sparse_support(
        (0,),
        ((1, 0),),
        values=(1.0,),
    )
    targets = target_model(x)

    optimizer = tf.keras.optimizers.Adam(0.01)
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean((model(x) - targets) ** 2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    optimizer_slots = [
        variable
        for variable in optimizer.variables
        if tuple(variable.shape) == (2,)
    ]
    for variable in optimizer_slots:
        variable.assign((3.0, 4.0))

    controller = AdaptiveWaveletSupport(
        layer,
        replacement_fraction=0.5,
        candidate_multiplier=2,
        seed=9,
    )
    result = controller.refresh(
        model,
        x,
        targets,
        tf.keras.losses.MeanSquaredError(),
        optimizer=optimizer,
    )

    replaced_position = result["replaced_positions"][0]
    assert result["grown_support"] == ((1, 0),)
    assert (1, 0) in layer.adaptive_sparse_support()
    assert layer.sparse_kernel_values[replaced_position] == 0
    assert int(layer.adaptive_sparse_update_count.numpy()) == 1
    assert result["optimizer_slots_reset"] == 2
    for variable in optimizer_slots:
        assert variable[replaced_position] == 0


def test_adaptive_controller_operates_inside_a_convolutional_model():
    inputs = tf.keras.Input((8, 1))
    features = tf.keras.layers.Conv1D(2, 1, use_bias=False)(inputs)
    layer = Volterra1D(
        m=2,
        filters=2,
        Ny=8,
        wave="bior2.2",
        backend="filterbank",
        level=1,
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=13,
        sparse_seed=51,
    )
    model = tf.keras.Model(inputs, layer(features))
    x = tf.random.stateless_normal((3, 8, 1), seed=(13, 17))
    targets = tf.random.stateless_normal((3, 8, 2), seed=(19, 23))
    before_shape = model(x).shape

    controller = AdaptiveWaveletSupport(
        layer,
        replacement_fraction=0.25,
        candidate_multiplier=2,
        seed=53,
    )
    result = controller.refresh(
        model,
        x,
        targets,
        tf.keras.losses.MeanSquaredError(),
    )

    assert model(x).shape == before_shape
    assert len(layer.adaptive_sparse_support()) == 13
    assert len(result["replaced_positions"]) == 4
    assert int(layer.adaptive_sparse_update_count.numpy()) == 1


def test_adaptive_support_rejects_noninteger_positions():
    layer = Volterra1D(
        m=1,
        filters=1,
        Ny=2,
        wave="haar",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=2,
    )
    layer(tf.zeros((1, 2, 1)))
    replacement = layer.sample_inactive_sparse_support(1)

    with pytest.raises(ValueError, match="positions must be"):
        layer.replace_adaptive_sparse_support((0.5,), replacement)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "wave": None,
                "kernel_parameterization": "sparse_wavelet",
                "sparse_support": "adaptive",
                "sparse_budget": 3,
            },
            "requires a wavelet name",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_support": "unknown",
                "sparse_budget": 3,
            },
            "sparse_support must be",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_support": "adaptive",
            },
            "sparse_budget must be",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_support": "adaptive",
                "sparse_budget": 3,
                "sparse_indices": [[0, 0, 0, 0]],
            },
            "sparse_indices cannot be supplied",
        ),
        (
            {
                "kernel_parameterization": "sparse_wavelet",
                "sparse_support": "fixed",
                "sparse_budget": 3,
                "sparse_indices": [[0, 0, 0, 0]],
            },
            "sparse_budget is only used",
        ),
    ],
)
def test_adaptive_sparse_arguments_are_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Volterra1D(**kwargs)


def test_adaptive_sparse_budget_is_checked_after_build():
    layer = Volterra1D(
        m=1,
        filters=1,
        Ny=2,
        wave="haar",
        kernel_parameterization="sparse_wavelet",
        sparse_support="adaptive",
        sparse_budget=5,
    )
    with pytest.raises(ValueError, match="sparse_budget exceeds"):
        layer(tf.zeros((1, 2, 1)))
