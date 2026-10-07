import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from ._wavelet_ops import (
    natural_to_wavelet_kernel,
    validate_backend,
    wavelet_to_natural_kernel,
)

#%% Shift variant Linear (m=1) Multiresolution Volterra Kernel
@tf.keras.utils.register_keras_serializable()
class LinearVolterra1D(tf.keras.layers.Layer):
    """
    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
    
    Linear (m=1) multiresolution Volterra Kernel (shift variant)
    Input: sequence x[n]
    Output: Linear monomial y1 (shift variant)

    --@KKT, 03Jul2025"""
    
    def __init__(self, filters=1, Ny=16, wave='haar', backend='matrix', **kwargs):
        super().__init__(**kwargs)
        # self.L = kernel_size
        self.wave = wave
        if self.wave is None: self.mra = False
        else: self.mra = True
        self.filters = filters
        self.Ny = Ny
        self.backend = validate_backend(backend)
        # self.mra = mra
    
    def build(self, input_shape):
        # input_shape: (batch_size, N, N, channels)
        self.N = input_shape[1]
        # self.L = self.N
        self.channels = input_shape[-1]
        kernel_shape = (self.Ny, self.N, input_shape[-1], self.filters)
        # kernel_shape = (self.L, input_shape[-1])
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
                    order=1,
                )
            initializer = initialize_wavelet_kernel

        # The stored kernel is h in natural mode and H in wavelet mode.
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer=initializer,
            trainable=True,
            # regularizer=self.kernel_regularizer,
            name='kernel'
        )
        # n = tf.range(self.N)
        # self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)

        super().build(input_shape)

    def call(self, x):  
        self.x = x  
        if self.mra==True:
            return self.__compute_output_with_mra_kernel(x)
        else: 
            return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        HT = self.__make_H()
        # print(HT.shape,'HT')
        ## l-axis below --->l and ^v
        
        ## H[v,l] 
        α = DWT1D(self.wave, clean=False, backend=self.backend)(x)
        # print('α = ', α.shape)
        # print('HT and α', HT.shape, α.shape)
        β = tf.einsum('bic,uico->buo',α, HT) #HT and α (16, 4, 1, 1) (None, 4, 1)
        # β = tf.einsum('buco->buo',β)
        # print('β', β.shape)

        # HTα = HT * α
        # print(HTα.shape, '\n', tf.squeeze(HTα).numpy())
        # beta = tf.expand_dims(tf.einsum('ijk->ik',HTα), axis=-1)
        
        # βT = tf.transpose(beta, perm=[1,0,2])
        # print('βT',tf.squeeze(βT).numpy())
        y = IDWT1D(self.wave, clean=False, backend=self.backend)(β)#
        # print('y',tf.squeeze(y).numpy())
        return y
         
    def __compute_output_in_natural_domain(self, x):
        h = self.__make_h()
        ynatural = tf.einsum('bic,uico->buo', x, h)
        # print(ynatural.shape, 'natural y') 
        return ynatural

    def sanity_check(self):
        return self.__compute_output_with_mra_kernel(self.x),  self.__compute_output_in_natural_domain(self.x)  
    
    def __make_H(self):
        return self.h1

    def __make_h(self):
        if not self.mra:
            return self.h1
        return wavelet_to_natural_kernel(
            self.h1,
            self.wave,
            order=1,
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



if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    ## Example
    # Functional model
    N, channels = 4, 1
    input_shape = (N, channels) 
    inputs = tf.keras.Input(shape=input_shape)
    filters, Ny = 1, 16
    H = LinearVolterra1D(filters=filters, Ny=Ny, wave='haar')
    # H = LinearVolterra1D(filters=filters, Ny=Ny, wave='haar', mra=False)
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 1D
    inputs_data = tf.random.normal((1, N, 1))
    targets = tf.random.normal((1, Ny, 1))
    # Training loop for 5 epochs
    epochs=5
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
