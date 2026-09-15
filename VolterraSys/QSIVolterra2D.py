import tensorflow as tf
# from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWT2DFB import DWT2D, IDWT2D
from VolterraSys.QSIVolterraNDlayout import QSIVolterraNDlayout

@tf.keras.utils.register_keras_serializable()
class QSIVolterra2D(QSIVolterraNDlayout):
    """ Quadratic (m=2) shift invariant multiresolution Volterra kernel for images (2D)

    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
    
    --kkt@3Jul2025"""
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, wave=wave, **kwargs)
        # Add any other initialization you need here 
    
    def make_h2_shifted_mra_kernel(self):
        self.h2 = self._make_h2()
        h2_shifted = self.make_h2_shifted_kernel()        
        # print('h2shifted', h2_shifted.shape)           
        shape = tf.shape(h2_shifted) # ij kl mn co 
        _ = tf.reshape(h2_shifted, [shape[0]*shape[1]*shape[2]*shape[3], shape[4], shape[5], shape[6]*shape[7]])
        _ = DWT2D(self.wave, clean=False)(_) #mn
        _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
        _ = tf.reshape(_, [shape[0]*shape[1], shape[2], shape[3], shape[4]*shape[5]*shape[6]*shape[7]])
        _ = DWT2D(self.wave, clean=False)(_) #kl
        _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
        _ = tf.reshape(_, [shape[0], shape[1], shape[2]*shape[3], shape[4]*shape[5]*shape[6]*shape[7]])
        _ = tf.transpose(_, perm=[2,0,1,3])
        _ = DWT2D(self.wave, clean=False)(_) #ij
        _ = tf.transpose(_, perm=[1,2,0,3])
        H = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            # return H
        # H = make_mra_kernel2(h2_shifted)
        # print('H',H.shape)
        return H

    def call(self, x):
        self.x = x
        if self.mra==True: return self._call_compute_with_mra_kernel(x)   
        else: return self._call_compute_in_natural_domain(x)

    def _call_compute_with_mra_kernel(self, x):
        x2_eq, y2_eq = self.get_einsum_strings(self.dim)
        x2 = tf.einsum(x2_eq, x, x)  
        # self.x2 = tf.einsum('bijc,bklc->bijklc', x, x) 
        def dwt4(x2):
            shape = tf.shape(x2)
            _ = tf.reshape(x2, [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5]])
            # print('_',_.shape)
            _ = DWT2D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]])
            _ = tf.transpose(_, perm=[0,3,4,1,2,5])
            _ = tf.reshape(_, [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5]])
            _ = DWT2D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]])
            return tf.transpose(_, perm=[0,3,4,1,2,5])
        α2 = dwt4(x2)
        # print('α2', α2.shape)
        H = self.make_h2_shifted_mra_kernel()
        ## Fitlering
        β = tf.einsum('ijklmnco, bklmnc->bijo', H, α2)
        # print('β', β.shape)

        # print(x2.shape, self.h2_shifted.shape)
        # y = tf.einsum('ijkco, bjkc->bio', self.h2_shifted, x2)
        # y2 = self.idwt1(β)
        y2 = IDWT2D(self.wave, clean=False)(β)
        # print('y2', y2.shape)
        return y2

    def sanity_check(self):
        return self._call_compute_with_mra_kernel(self.x), self._call_compute_in_natural_domain(self.x)  

        
if __name__ =='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    ## Example
    # Functional model
    N, channels, filters = 8, 1, 1
    input_shape = (N, N, channels)  # Replace N with the actual size of x       #2D
    inputs = tf.keras.Input(shape=input_shape)
    # Apply the custom layer to the inputs
    # H = QSIVolterra2D(filters=filters, wave=None)
    H = QSIVolterra2D(filters=filters, wave='db2')
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 2D Random data
    x = tf.random.normal((1, N, N, 1))
    y = tf.random.normal((1, N, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(x, y, epochs=5, verbose=1)    