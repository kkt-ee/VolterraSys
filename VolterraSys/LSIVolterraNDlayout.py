import tensorflow as tf
# import keras
# from TFDWT.DWT1DFB import DWT1D, IDWT1D

@tf.keras.utils.register_keras_serializable()
class LSIVolterraNDlayout(tf.keras.layers.Layer):
    """ Linear shift invariant multiresolution Volterra kernel layout for D-dimensional data 
    (present support 1D, 2D, 3D)

    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
        
    Linear (m=1) shift invariant multiresolution Volterra kernel layout for D-dimensional input
    Input: general D-dimensional signal x[n1,n2,...,nD]
    Output: Linear shift invariant D-dimensional monomial y1[n1,n2,...,nD]

    --@KKT@04Jul2025 """
    
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.filters = filters
        self.wave = wave
        if self.wave is None: self.mra = False
        else: self.mra = True

    def build(self, input_shape):
        self.N = input_shape[1]         # sequence length
        self.axes = list(range(1, len(input_shape) - 1))
        self.dim = self.axes[-1]
        self.channels = input_shape[-1] # number of channels
        # Multi-channel kernel: (L, channels)
        # kernel_shape = (self.L, self.channels, self.filters)
        kernel_shape = [self.L] * self.dim + [self.channels, self.filters]
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            name='UIR_kernel')
        super().build(input_shape)
    
    def call(self, x):
        pass
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.L,
            'wave': self.wave,
            'filters': self.filters,
        })
        return config
