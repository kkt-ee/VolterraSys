import string
from itertools import combinations_with_replacement
from math import factorial
from numbers import Integral

import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.multilevel.dwt import dwt_packed_axis, idwt_packed_axis
from ._wavelet_ops import (
    analysis_axis,
    analysis_transpose_axis,
    natural_to_wavelet_kernel,
    synthesis_axis,
    synthesis_transpose_axis,
    validate_backend,
    validate_level,
    wavelet_to_natural_kernel,
)


def _normalize_sparse_indices(sparse_indices, width):
    if sparse_indices is None:
        raise ValueError(
            "sparse_indices must be provided when "
            "kernel_parameterization='sparse_wavelet'."
        )
    try:
        rows = tuple(tuple(row) for row in sparse_indices)
    except TypeError as error:
        raise ValueError(
            "sparse_indices must be a sequence of index rows."
        ) from error
    if not rows:
        raise ValueError("sparse_indices must contain at least one row.")

    normalized = []
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"Each sparse index row must contain {width} entries."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in row
        ):
            raise ValueError("sparse_indices entries must be integers.")
        normalized_row = tuple(int(value) for value in row)
        if any(value < 0 for value in normalized_row):
            raise ValueError("sparse_indices entries must be nonnegative.")
        normalized.append(normalized_row)

    normalized = tuple(normalized)
    if len(set(normalized)) != len(normalized):
        raise ValueError("sparse_indices rows must be unique.")
    return normalized


def _canonical_input_tuples(input_size, order):
    """Return all nondecreasing flattened input-index tuples."""
    return tuple(combinations_with_replacement(range(input_size), order))


def _permutation_multiplicity(input_indices):
    """Return the number of distinct permutations of one canonical tuple."""
    multiplicity = factorial(len(input_indices))
    run_length = 1
    for previous, current in zip(
        input_indices[:-1],
        input_indices[1:],
    ):
        if current == previous:
            run_length += 1
        else:
            multiplicity //= factorial(run_length)
            run_length = 1
    return multiplicity // factorial(run_length)


def _unique_permutations(items):
    """Yield the distinct permutations of a sorted tuple."""
    if not items:
        yield ()
        return
    for index, item in enumerate(items):
        if index and item == items[index - 1]:
            continue
        remaining = items[:index] + items[index + 1:]
        for suffix in _unique_permutations(remaining):
            yield (item,) + suffix


# %% Shift-variant mth-order multiresolution Volterra kernel
@tf.keras.utils.register_keras_serializable()
class Volterra1D(tf.keras.layers.Layer):
    """General mth-order multiresolution Volterra kernel (shift variant).

    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Input: sequence x[n]
    Output: mth-order monomial ym (shift variant)

    kernel_parameterization='symmetric' stores every input-symmetric kernel
    coefficient once in canonical order. No support indices are supplied.

    With kernel_parameterization='sparse_wavelet', sparse_indices contains
    fixed rows [output, input_1, ..., input_m, channel_1, ..., channel_m,
    filter]. Only sparse_kernel_values is trained. No automatic thresholding
    or dense kernel construction occurs during the forward computation.

    --@KKT, 06Oct2026
    """

    def __init__(
        self,
        m=1,
        filters=1,
        Ny=16,
        wave='haar',
        backend='matrix',
        level=1,
        kernel_parameterization='dense',
        kernel_rank=None,
        sparse_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("m must be a positive integer.")

        self.m = m
        self.wave = wave
        self.mra = self.wave is not None
        self.filters = filters
        self.Ny = Ny
        self.level = validate_level(level)
        self.backend = validate_backend(backend)
        if kernel_parameterization not in (
            'dense',
            'symmetric',
            'symmetric_cp',
            'sparse_wavelet',
        ):
            raise ValueError(
                "kernel_parameterization must be 'dense', "
                "'symmetric', 'symmetric_cp', or 'sparse_wavelet'."
            )
        if kernel_parameterization != 'symmetric_cp' and (
            kernel_rank is not None
        ):
            raise ValueError(
                "kernel_rank is only used with "
                "kernel_parameterization='symmetric_cp'."
            )
        if kernel_parameterization != 'sparse_wavelet' and (
            sparse_indices is not None
        ):
            raise ValueError(
                "sparse_indices is only used with "
                "kernel_parameterization='sparse_wavelet'."
            )
        if kernel_parameterization == 'symmetric_cp' and (
            isinstance(kernel_rank, bool)
            or not isinstance(kernel_rank, int)
            or kernel_rank < 1
        ):
            raise ValueError(
                "kernel_rank must be a positive integer when "
                "kernel_parameterization='symmetric_cp'."
            )
        if kernel_parameterization == 'sparse_wavelet':
            if not self.mra:
                raise ValueError(
                    "kernel_parameterization='sparse_wavelet' requires "
                    "a wavelet name."
                )
            sparse_indices = _normalize_sparse_indices(
                sparse_indices,
                width=2 * self.m + 2,
            )
        self.kernel_parameterization = kernel_parameterization
        self.kernel_rank = kernel_rank
        self.sparse_indices = sparse_indices

        self._make_einsum_equations()

    def _make_einsum_equations(self):
        labels = [
            label for label in string.ascii_letters
            if label not in 'buor'
        ]
        if 3 * self.m > len(labels):
            raise ValueError("m is too large for a single vectorized einsum.")

        input_indices = labels[:self.m]
        channel_indices = labels[self.m:2 * self.m]
        coefficient_indices = labels[2 * self.m:3 * self.m]

        input_terms = [
            f'b{input_index}{channel_index}'
            for input_index, channel_index in zip(
                input_indices, channel_indices
            )
        ]
        kernel_term = (
            'u'
            + ''.join(input_indices)
            + ''.join(channel_indices)
            + 'o'
        )
        self._contraction_equation = (
            ','.join(input_terms + [kernel_term]) + '->buo'
        )

        synthesis_terms = [
            input_index + coefficient_index
            for input_index, coefficient_index in zip(
                input_indices, coefficient_indices
            )
        ]
        transformed_kernel_term = (
            'u'
            + ''.join(coefficient_indices)
            + ''.join(channel_indices)
            + 'o'
        )
        self._kernel_transform_equation = (
            ','.join([kernel_term] + synthesis_terms)
            + '->'
            + transformed_kernel_term
        )

        factor_terms = [
            input_index + channel_index + 'r'
            for input_index, channel_index in zip(
                input_indices, channel_indices
            )
        ]
        self._factor_materialization_equation = (
            ','.join(['uor'] + factor_terms)
            + '->'
            + kernel_term
        )

    def build(self, input_shape):
        if input_shape[1] is None or input_shape[-1] is None:
            raise ValueError("The input length and channel count must be known.")

        self.N = int(input_shape[1])
        self.channels = int(input_shape[-1])

        # hm[output, input_1, ..., input_m,
        #    channel_1, ..., channel_m, filter].
        kernel_shape = (
            (self.Ny,)
            + (self.N,) * self.m
            + (self.channels,) * self.m
            + (self.filters,)
        )
        self._kernel_shape = kernel_shape

        if self.kernel_parameterization == 'dense':
            initializer = 'glorot_uniform'
            if self.mra:
                def initialize_wavelet_kernel(shape, dtype=None):
                    natural_kernel = tf.keras.initializers.GlorotUniform()(
                        shape,
                        dtype=dtype,
                    )
                    return natural_to_wavelet_kernel(
                        natural_kernel,
                        self.wave,
                        order=self.m,
                        level=self.level,
                    )
                initializer = initialize_wavelet_kernel

            # The stored kernel is h in natural mode and H in wavelet mode.
            self.hm = self.add_weight(
                shape=kernel_shape,
                initializer=initializer,
                trainable=True,
                name='kernel',
            )
        elif self.kernel_parameterization == 'symmetric':
            self._symmetric_input_tuples = _canonical_input_tuples(
                self.N * self.channels,
                self.m,
            )
            with tf.init_scope():
                self._symmetric_input_indices = tuple(
                    tf.constant(
                        [
                            indices[mode]
                            for indices in self._symmetric_input_tuples
                        ],
                        dtype=tf.int32,
                    )
                    for mode in range(self.m)
                )
                self._symmetric_multiplicities = tf.constant(
                    [
                        _permutation_multiplicity(indices)
                        for indices in self._symmetric_input_tuples
                    ],
                    dtype=self.compute_dtype,
                )
            self.symmetric_kernel_values = self.add_weight(
                shape=(
                    self.Ny,
                    len(self._symmetric_input_tuples),
                    self.filters,
                ),
                initializer='glorot_uniform',
                trainable=True,
                name='kernel',
            )
        elif self.kernel_parameterization == 'symmetric_cp':
            input_factor_shape = (
                self.N,
                self.channels,
                self.kernel_rank,
            )
            output_factor_shape = (
                self.Ny,
                self.filters,
                self.kernel_rank,
            )
            input_initializer = 'glorot_uniform'
            output_initializer = 'glorot_uniform'
            if self.mra:
                def initialize_wavelet_input_factors(shape, dtype=None):
                    natural_factors = tf.keras.initializers.GlorotUniform()(
                        shape,
                        dtype=dtype,
                    )
                    return synthesis_transpose_axis(
                        natural_factors,
                        self.wave,
                        axis=0,
                        level=self.level,
                    )

                def initialize_wavelet_output_factors(shape, dtype=None):
                    natural_factors = tf.keras.initializers.GlorotUniform()(
                        shape,
                        dtype=dtype,
                    )
                    return analysis_axis(
                        natural_factors,
                        self.wave,
                        axis=0,
                        level=self.level,
                    )

                input_initializer = initialize_wavelet_input_factors
                output_initializer = initialize_wavelet_output_factors

            self.input_factors = self.add_weight(
                shape=input_factor_shape,
                initializer=input_initializer,
                trainable=True,
                name='input_factors',
            )
            self.output_factors = self.add_weight(
                shape=output_factor_shape,
                initializer=output_initializer,
                trainable=True,
                name='output_factors',
            )
        else:
            for row in self.sparse_indices:
                for axis, (index, limit) in enumerate(
                    zip(row, kernel_shape)
                ):
                    if index >= limit:
                        raise ValueError(
                            "sparse_indices contains index "
                            f"{index} outside axis {axis} with size {limit}."
                        )

            # Keep fixed indices available if build is first called while
            # tracing a TensorFlow function.
            with tf.init_scope():
                self._sparse_input_indices = tuple(
                    tf.constant(
                        [
                            (row[1 + mode], row[1 + self.m + mode])
                            for row in self.sparse_indices
                        ],
                        dtype=tf.int32,
                    )
                    for mode in range(self.m)
                )
                self._sparse_output_segments = tf.constant(
                    [
                        row[0] * self.filters + row[-1]
                        for row in self.sparse_indices
                    ],
                    dtype=tf.int32,
                )
            self.sparse_kernel_values = self.add_weight(
                shape=(len(self.sparse_indices),),
                initializer='glorot_uniform',
                trainable=True,
                name='kernel_values',
            )

        if self.mra and self.backend == 'matrix':
            def initialize_analysis(length):
                def initializer(shape, dtype=None):
                    if self.level > 1:
                        return dwt_packed_axis(
                            tf.eye(length, dtype=dtype or tf.float32),
                            level=self.level,
                            wave=self.wave,
                            axis=0,
                            backend='matrix',
                        )
                    dwt = DWT1D(self.wave, clean=False)
                    dwt.build((None, length, 1))
                    return tf.cast(dwt.A, dtype or tf.float32)
                return initializer

            def initialize_synthesis(length):
                def initializer(shape, dtype=None):
                    if self.level > 1:
                        return idwt_packed_axis(
                            tf.eye(length, dtype=dtype or tf.float32),
                            level=self.level,
                            wave=self.wave,
                            axis=0,
                            backend='matrix',
                        )
                    idwt = IDWT1D(self.wave, clean=False)
                    idwt.build((None, length, 1))
                    return tf.cast(idwt.S, dtype or tf.float32)
                return initializer

            self.A_input = self.add_weight(
                name='input_analysis',
                shape=(self.N, self.N),
                initializer=initialize_analysis(self.N),
                trainable=False,
            )
            self.S_output = self.add_weight(
                name='output_synthesis',
                shape=(self.Ny, self.Ny),
                initializer=initialize_synthesis(self.Ny),
                trainable=False,
            )

        super().build(input_shape)

    def call(self, x):
        self.x = x
        if self.mra:
            return self.__compute_output_with_mra_kernel(x)
        return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        if self.backend == 'matrix':
            alpha = tf.einsum('li,bic->blc', self.A_input, x)
        else:
            alpha = analysis_axis(
                x,
                self.wave,
                axis=1,
                level=self.level,
            )
        if self.kernel_parameterization == 'dense':
            beta = tf.einsum(
                self._contraction_equation,
                *([alpha] * self.m),
                self.__make_H(),
            )
        elif self.kernel_parameterization == 'symmetric':
            beta = self._contract_symmetric(alpha)
        elif self.kernel_parameterization == 'symmetric_cp':
            beta = self._contract_factorized(
                alpha,
                self.input_factors,
                self.output_factors,
            )
        else:
            beta = self._contract_sparse(alpha)
        if self.backend == 'matrix':
            return tf.einsum('uv,bvo->buo', self.S_output, beta)
        return synthesis_axis(
            beta,
            self.wave,
            axis=1,
            level=self.level,
        )

    def __compute_output_in_natural_domain(self, x):
        if self.kernel_parameterization == 'dense':
            return tf.einsum(
                self._contraction_equation,
                *([x] * self.m),
                self.__make_h(),
            )
        if self.kernel_parameterization == 'symmetric':
            if not self.mra:
                return self._contract_symmetric(x)
            return tf.einsum(
                self._contraction_equation,
                *([x] * self.m),
                self.__make_h(),
            )
        if self.kernel_parameterization == 'sparse_wavelet':
            return tf.einsum(
                self._contraction_equation,
                *([x] * self.m),
                self.__make_h(),
            )
        input_factors, output_factors = self._natural_factors()
        return self._contract_factorized(
            x,
            input_factors,
            output_factors,
        )

    def _contract_factorized(
        self,
        x,
        input_factors,
        output_factors,
    ):
        projected = tf.einsum('bic,icr->br', x, input_factors)
        monomials = tf.pow(projected, self.m)
        return tf.einsum('br,uor->buo', monomials, output_factors)

    def _contract_symmetric(self, coordinates):
        """Contract one canonical coefficient per input-permutation orbit."""
        flattened_input = tf.reshape(
            coordinates,
            (-1, self.N * self.channels),
        )
        monomials = None
        for indices in self._symmetric_input_indices:
            values = tf.gather(flattened_input, indices, axis=1)
            if monomials is None:
                monomials = values
            else:
                monomials = monomials * values
        monomials = monomials * self._symmetric_multiplicities[None, :]
        return tf.einsum(
            'bt,utf->buf',
            monomials,
            self.symmetric_kernel_values,
        )

    def _contract_sparse(self, alpha):
        """Contract only active entries of H without a dense kernel tensor."""
        alpha_by_coordinate = tf.transpose(alpha, (1, 2, 0))
        monomials = None
        for coordinates in self._sparse_input_indices:
            values = tf.transpose(
                tf.gather_nd(alpha_by_coordinate, coordinates)
            )
            if monomials is None:
                monomials = values
            else:
                monomials = monomials * values

        contributions = monomials * self.sparse_kernel_values[None, :]
        segment_totals = tf.math.unsorted_segment_sum(
            tf.transpose(contributions),
            self._sparse_output_segments,
            self.Ny * self.filters,
        )
        beta = tf.reshape(
            segment_totals,
            (self.Ny, self.filters, -1),
        )
        return tf.transpose(beta, (2, 0, 1))

    def _natural_factors(self):
        if not self.mra:
            return self.input_factors, self.output_factors
        return (
            analysis_transpose_axis(
                self.input_factors,
                self.wave,
                axis=0,
                level=self.level,
            ),
            synthesis_axis(
                self.output_factors,
                self.wave,
                axis=0,
                level=self.level,
            ),
        )

    def _materialize_factors(self, input_factors, output_factors):
        return tf.einsum(
            self._factor_materialization_equation,
            output_factors,
            *([input_factors] * self.m),
        )

    def materialize_kernel(self, domain=None):
        """Materialize the implied dense kernel for inspection or testing."""
        if not self.built:
            raise ValueError("The layer must be built before materialization.")
        if domain is None:
            domain = 'wavelet' if self.mra else 'natural'
        if domain not in ('natural', 'wavelet'):
            raise ValueError("domain must be 'natural' or 'wavelet'.")
        if domain == 'wavelet' and not self.mra:
            raise ValueError(
                "A natural-domain layer has no stored wavelet kernel."
            )
        if domain == 'wavelet':
            return self.__make_H()
        return self.__make_h()

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        if self.kernel_parameterization == 'symmetric':
            return self._materialize_symmetric_kernel()
        if self.kernel_parameterization == 'symmetric_cp':
            return self._materialize_factors(
                self.input_factors,
                self.output_factors,
            )
        if self.kernel_parameterization == 'sparse_wavelet':
            # Flatten only for this diagnostic. TensorFlow ScatterNd does not
            # support arbitrary-rank index rows on every backend.
            kernel_size = 1
            for length in self._kernel_shape:
                kernel_size *= length
            if kernel_size > 2**63 - 1:
                raise ValueError(
                    "The implied dense kernel is too large to materialize."
                )
            flat_indices = []
            for row in self.sparse_indices:
                flat_index = 0
                for index, limit in zip(row, self._kernel_shape):
                    flat_index = flat_index * limit + index
                flat_indices.append((flat_index,))
            flat_kernel = tf.scatter_nd(
                tf.constant(flat_indices, dtype=tf.int64),
                self.sparse_kernel_values,
                tf.constant((kernel_size,), dtype=tf.int64),
            )
            return tf.reshape(flat_kernel, self._kernel_shape)
        return self.hm

    def _materialize_symmetric_kernel(self):
        """Materialize the exact symmetric tensor for diagnostics."""
        kernel_size = 1
        for length in self._kernel_shape:
            kernel_size *= length
        if kernel_size > 2**63 - 1:
            raise ValueError(
                "The implied dense kernel is too large to materialize."
            )

        flat_indices = []
        packed_indices = []
        tuple_count = len(self._symmetric_input_tuples)
        for tuple_index, input_indices in enumerate(
            self._symmetric_input_tuples
        ):
            for permutation in _unique_permutations(input_indices):
                for output_index in range(self.Ny):
                    for filter_index in range(self.filters):
                        input_coordinates = tuple(
                            divmod(index, self.channels)
                            for index in permutation
                        )
                        row = (
                            (output_index,)
                            + tuple(
                                position
                                for position, _ in input_coordinates
                            )
                            + tuple(
                                channel
                                for _, channel in input_coordinates
                            )
                            + (filter_index,)
                        )
                        flat_index = 0
                        for index, limit in zip(row, self._kernel_shape):
                            flat_index = flat_index * limit + index
                        flat_indices.append((flat_index,))
                        packed_indices.append(
                            (
                                output_index * tuple_count
                                + tuple_index
                            ) * self.filters
                            + filter_index
                        )
        flat_kernel = tf.scatter_nd(
            tf.constant(flat_indices, dtype=tf.int64),
            tf.gather(
                tf.reshape(self.symmetric_kernel_values, (-1,)),
                packed_indices,
            ),
            tf.constant((kernel_size,), dtype=tf.int64),
        )
        return tf.reshape(flat_kernel, self._kernel_shape)

    def __make_h(self):
        if self.kernel_parameterization == 'symmetric_cp':
            input_factors, output_factors = self._natural_factors()
            return self._materialize_factors(
                input_factors,
                output_factors,
            )
        if not self.mra:
            if self.kernel_parameterization == 'symmetric':
                return self.__make_H()
            return self.hm
        return wavelet_to_natural_kernel(
            self.__make_H(),
            self.wave,
            order=self.m,
            level=self.level,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'm': self.m,
            'Ny': self.Ny,
            'wave': self.wave,
            'level': self.level,
            'filters': self.filters,
            'backend': self.backend,
            'kernel_parameterization': self.kernel_parameterization,
            'kernel_rank': self.kernel_rank,
            'sparse_indices': (
                None
                if self.sparse_indices is None
                else [list(row) for row in self.sparse_indices]
            ),
        })
        return config


if __name__ == '__main__':
    N, channels = 4, 1
    filters, Ny, m = 1, 16, 4
    inputs = tf.keras.Input(shape=(N, channels))
    layer = Volterra1D(m=m, filters=filters, Ny=Ny, wave='haar')
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
