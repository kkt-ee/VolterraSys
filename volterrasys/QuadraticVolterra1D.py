import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from ._wavelet_ops import (
    natural_to_wavelet_kernel,
    validate_backend,
    wavelet_to_natural_kernel,
)


# %% Shift variant Quadratic (m=2) Multiresolution Volterra Kernel
@tf.keras.utils.register_keras_serializable()
class QuadraticVolterra1D(tf.keras.layers.Layer):
    """Quadratic (m=2) multiresolution Volterra kernel (shift variant).

    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Input: sequence x[n]
    Output: Quadratic monomial y2 (shift variant)

    --@KKT, 06Oct2026
    """

    def __init__(self, filters=1, Ny=16, wave='haar', backend='matrix', **kwargs):
        super().__init__(**kwargs)
        self.wave = wave
        if self.wave is None:
            self.mra = False
        else:
            self.mra = True
        self.filters = filters
        self.Ny = Ny
        self.backend = validate_backend(backend)

    def build(self, input_shape):
        # input_shape: (batch_size, N, channels)
        self.N = input_shape[1]
        self.channels = input_shape[-1]

        # h2[output, input_1, input_2, channel_1, channel_2, filter].
        kernel_shape = (
            self.Ny,
            self.N,
            self.N,
            self.channels,
            self.channels,
            self.filters,
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
                    order=2,
                )
            initializer = initialize_wavelet_kernel

        # The stored kernel is h in natural mode and H in wavelet mode.
        self.h2 = self.add_weight(
            shape=kernel_shape,
            initializer=initializer,
            trainable=True,
            name='kernel',
        )

        super().build(input_shape)

    def call(self, x):
        self.x = x
        if self.mra is True:
            return self.__compute_output_with_mra_kernel(x)
        else:
            return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        H = self.__make_H()
        alpha = DWT1D(self.wave, clean=False, backend=self.backend)(x)
        beta = tf.einsum('bic,bjd,uijcdo->buo', alpha, alpha, H)
        y = IDWT1D(self.wave, clean=False, backend=self.backend)(beta)
        return y

    def __compute_output_in_natural_domain(self, x):
        h = self.__make_h()
        ynatural = tf.einsum('bic,bjd,uijcdo->buo', x, x, h)
        return ynatural

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        return self.h2

    def __make_h(self):
        if not self.mra:
            return self.h2
        return wavelet_to_natural_kernel(
            self.h2,
            self.wave,
            order=2,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'Ny': self.Ny,
            'wave': self.wave,
            'filters': self.filters,
            'backend': self.backend,
        })
        return config


if __name__ == '__main__':
    N, channels = 4, 1
    filters, Ny = 1, 16
    inputs = tf.keras.Input(shape=(N, channels))
    layer = QuadraticVolterra1D(filters=filters, Ny=Ny, wave='haar')
    outputs = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()

    ## 1D
    inputs_data = tf.random.normal((1, N, 1))
    targets = tf.random.normal((1, Ny, 1))
    # Training loop for 5 epochs
    epochs=5
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
