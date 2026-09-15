import tensorflow as tf
# import keras
from TFDWT.DWT2DFB import DWT2D, IDWT2D
from VolterraSys.LSIVolterraNDlayout import LSIVolterraNDlayout

@tf.keras.utils.register_keras_serializable()
class LSIVolterra2D(LSIVolterraNDlayout):
    """ 
    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
    
    Linear (m=1) shift invariant multiresolution Volterra Kernel for images
    Input: image x[n1,n2] (rectangular image)
    Output: Linear shift invariant monomial y1[n1,n2]

    --@KKT@04Jul2025"""
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, wave=wave, **kwargs)

    def __makeH(self):
        # Npoint = self.N
        paddings = [
        [0, self.N - self.L],  # height dimension
        [0, self.N - self.L],  # width dimension
        [0,0],
        [0,0] 
        ]
        h1 = tf.pad(self.h1, paddings, mode='CONSTANT', constant_values=0)

        # Create a meshgrid for n1, n2, k1, k2
        n1 = tf.range(self.N)  # [0, 1, ..., N-1]
        n2 = tf.range(self.N)  # [0, 1, ..., N-1]
        k1 = tf.range(self.N)  # [0, 1, ..., N-1]
        k2 = tf.range(self.N)  # [0, 1, ..., N-1]

        # Create a grid of indices for n1, n2, k1, k2
        n1, n2, k1, k2 = tf.meshgrid(n1, n2, k1, k2, indexing='ij')  # Shape (N, N, N, N)

        # Compute circularly shifted indices
        shifted_n1 = tf.math.mod(n1 - k1, self.N)  # Circular shift for n1
        shifted_n2 = tf.math.mod(n2 - k2, self.N)  # Circular shift for n2

        # Gather the values from h using the shifted indices
        h1_shifted = tf.gather_nd(h1, tf.stack([shifted_n1, shifted_n2], axis=-1))
        # h1_shifted = tf.transpose(h1_shifted,perm=[0,2,3,1,4,5])  # n1,:,:,n2 01234 ## very important line!!
        # print('++', h1_shifted.shape)
        self.h1_shifted = h1_shifted
        if self.mra == False: return self.h1_shifted
        else:               
            # ### second vectorized upgrade
            def dwtmra(h1_shifted):
                # # h1_shifted: (N, N, N, N, in_channels, out_channels)
                N, in_channels, out_channels = self.N, self.channels, self.filters
                num_io = in_channels * out_channels

                # Flatten (in, out) for DWT2D
                hflat = tf.reshape(h1_shifted, (N, N, N, N, num_io))
                hflat_reshape = tf.reshape(hflat, (N*N, N, N, num_io))

                dwt2d = DWT2D(self.wave, clean=False)
                x1 = dwt2d(tf.cast(hflat_reshape, tf.float32))
                x1r = tf.reshape(x1, (N, N, N, N, num_io))

                # Transpose for columnplanewise DWT2D
                x1T = tf.transpose(x1r, [2, 3, 0, 1, 4])
                x1T_reshape = tf.reshape(x1T, (N*N, N, N, num_io))
                x2 = dwt2d(x1T_reshape)
                x2r = tf.reshape(x2, (N, N, N, N, num_io))

                # Transpose back
                HT = tf.transpose(x2r, [2, 3, 0, 1, 4])
                HT = tf.reshape(HT, (N, N, N, N, in_channels, out_channels))
                # print('++', HT.shape)
                return HT
            return dwtmra(h1_shifted)   

    def call(self, x):
        self.x = x
        if self.mra==True: return self._call_with_mra_kernel(x)   
        else: return self._call_in_natural_domain(x)  

    def _call_with_mra_kernel(self, x):
        # HT = self.__getH()
        α = DWT2D(self.wave, clean=False)(tf.cast(x, dtype=tf.float32))
        # α.shape
        HT = self.__makeH()
        ## convolution like in wavelet domain
        # β = tf.einsum('bijc,uijvco->buvco', α, self.HT) #ok
        β = tf.einsum('bijc,uvijco->buvo', α, HT)
        # β = tf.einsum('buvco->buvo',β)
        # β
        y = IDWT2D(self.wave, clean=False)(β)
        # ynatural = tf.einsum('bijc,uijv->buvc', x, tf.cast(self.h_shifted, tf.float32))[0,:,:,0]
        return y#, ynatural

    def _call_in_natural_domain(self, x):
        h1_shifted = self.__makeH() ##dummy!!
        return tf.einsum('bijc,uvijco->buvo', x, self.h1_shifted)

    def sanity_check(self):
        return self._call_with_mra_kernel(self.x), self._call_in_natural_domain(self.x)  


if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    ## Example
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, N, channels) 
    inputs = tf.keras.Input(shape=input_shape)
    H = LSIVolterra2D(filters=1)
    H = LSIVolterra2D(filters=1, wave=None)
    H = LSIVolterra2D(filters=1, kernel_size=(M,N), wave=None)
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 2D
    inputs_data = tf.random.normal((1, N, N, 1))
    targets = tf.random.normal((1, N, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)