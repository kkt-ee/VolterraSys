import string

import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.multilevel.dwt import dwt_packed_axis, idwt_packed_axis
from ._wavelet_ops import (
    analysis_axis,
    natural_to_wavelet_kernel,
    synthesis_axis,
    validate_backend,
    validate_level,
    wavelet_to_natural_kernel,
)


# %% Shift-variant mth-order multiresolution Volterra kernel
@tf.keras.utils.register_keras_serializable()
class Volterra1D(tf.keras.layers.Layer):
    """General mth-order multiresolution Volterra kernel (shift variant).

    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Input: sequence x[n]
    Output: mth-order monomial ym (shift variant)

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

        self._make_einsum_equations()

    def _make_einsum_equations(self):
        labels = [label for label in string.ascii_letters if label not in 'buo']
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
        H = self.__make_H()
        if self.backend == 'matrix':
            alpha = tf.einsum('li,bic->blc', self.A_input, x)
        else:
            alpha = analysis_axis(
                x,
                self.wave,
                axis=1,
                level=self.level,
            )
        beta = tf.einsum(
            self._contraction_equation,
            *([alpha] * self.m),
            H,
        )
        if self.backend == 'matrix':
            return tf.einsum('uv,bvo->buo', self.S_output, beta)
        return synthesis_axis(
            beta,
            self.wave,
            axis=1,
            level=self.level,
        )

    def __compute_output_in_natural_domain(self, x):
        h = self.__make_h()
        return tf.einsum(
            self._contraction_equation,
            *([x] * self.m),
            h,
        )

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        return self.hm

    def __make_h(self):
        if not self.mra:
            return self.hm
        return wavelet_to_natural_kernel(
            self.hm,
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
