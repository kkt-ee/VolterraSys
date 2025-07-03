import tensorflow as tf
import keras
from TFDWT.DWT2DFB import DWT2D, IDWT2D

@tf.keras.utils.register_keras_serializable()
class LSIVolterra2Dmra(tf.keras.layers.Layer):
    """ VolterraSys: Multidimensional linear and nonlinear Volterra kernels in natural and multiresolution bases.
        Copyright (C) 2025 Kishore Kumar Tarafdar

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>. 
        
    
    Shift invariant Linear (m=1) Multiresolution Volterra Kernel
       Input: image x[n1,n2]
       Output: Shift invariant linear monomial y1, i.e., m=1

    --@KKT@03Jul2025"""
    
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.wave = wave
        self.filters = filters
    
    def build(self, input_shape):
        # input_shape: (batch_size, N, N, channels)
        self.N = input_shape[1]
        self.channels = input_shape[-1]

        kernel_shape = (self.L, self.L, input_shape[-1], self.filters)
        # kernel_shape = (self.L, self.L, input_shape[-1])
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            # regularizer=self.kernel_regularizer,
            name='kernel')
        super().build(input_shape)

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
        print('++', h1_shifted.shape)
       
        # ### second vectorized upgrade
        def dwtmra(h1_shited):
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
            return HT
        return dwtmra(h1_shifted)     
                    
            # PERFECT but slow    
            # # for in channels
            # h1_n_l = tf.stack([
            #     tf.stack([
            #         tf.transpose(
            #             DWT2D(self.wave, clean=False)(
            #                 tf.transpose(
            #                     DWT2D(self.wave, clean=False)(tf.cast(h1_shifted[...,i,j], dtype=tf.float32)), 
            #                         perm=[1,0,3,2])
            #                         ),
            #                         perm=[1,0,3,2]) for i in range(self.channels)
            #         ],axis=-1) for j in range(self.filters)
            #         ],axis=-1)
            
            # ## vectorized upgrade (perfect and fast)
            # def foo(h1_shifted):
            #     # Example placeholder tensor
            #     # h1_shifted shape: (N, N, N, N, in_channels, out_channels)
            #     # You can replace these with actual values
            #     N = self.N
            #     in_channels = self.channels
            #     out_channels = self.filters
            #     # h1_shifted = tf.random.normal((N, N, N, N, in_channels, out_channels))


            #     def reshape1(h1_shifted):
            #         ##### Step 1: Transpose to bring in_channels and out_channels to the front
            #         # New shape: (in_channels, out_channels, N, N, N, N)
            #         h1_transposed = tf.transpose(h1_shifted, perm=[4, 5, 0, 1, 2, 3])

            #         # Step 2: Reshape to (out_channels * in_channels * N, N, N, N)
            #         new_shape = (out_channels * in_channels * N, N, N, N)
            #         h1_reshaped = tf.reshape(h1_transposed, new_shape)
            #         print(h1_reshaped.shape)
            #         return h1_reshaped
        
            #     h1_reshaped = reshape1(h1_shifted)
            #     transform1 = DWT2D(self.wave, clean=False)(tf.cast(h1_reshaped, dtype=tf.float32))

            #     def reshape1back(transform1):
            #         #####get back to original shape
            #         # Step 1: Reshape back to (in_channels, out_channels, N, N, N, N)
            #         transform1_back = tf.reshape(transform1, (in_channels, out_channels, N, N, N, N))

            #         # Step 2: Transpose back to original shape: (N, N, N, N, in_channels, out_channels)
            #         transform1_back = tf.transpose(transform1_back, perm=[2, 3, 4, 5, 0, 1])
            #         print(transform1_back.shape)
            #         return transform1_back

            #     transform1_back = reshape1back(transform1)
            #     transform1_back = tf.transpose(transform1_back, perm=[1,0,3,2,4,5])

            #     transform1_back_reshaped = reshape1(transform1_back)
            #     transform2 = DWT2D(self.wave, clean=False)(tf.cast(transform1_back_reshaped, dtype=tf.float32))
            #     transform2_back = reshape1back(transform2)
            #     transform2_back = tf.transpose(transform2_back, perm=[1,0,3,2,4,5])
            #     return transform2_back

            # h1_n_l = foo(h1_shifted)
                    
        
            # # h1_n_lT = [tf.transpose(_, perm=[1,0,3,2]) for _ in h1_n_l]
            # # H = [DWT2D(self.wave)(_) for _ in h1_n_lT]
            # # HT = [tf.transpose(H, perm=[1,0,3,2]) for H in H]
            # # HT = tf.stack(HT, axis=-1)
            
            # HT = h1_n_l
            # print('>',HT.shape)
        
            # # h1_n_l = DWT2D(self.wave)(tf.cast(h1_shifted, dtype=tf.float32))
            # # h1_n_l.shape
            # # h1_n_lT = tf.transpose(h1_n_l, perm=[1,0,3,2])
            # # # h_initT[0,:,:,0]
            # # H = DWT2D(self.wave)(h1_n_lT)
            # # HT = tf.transpose(H, perm=[1,0,3,2])
            # return HT
    

    def call(self, x):

        # HT = self.__getH()

        α = DWT2D(self.wave, clean=False)(tf.cast(x, dtype=tf.float32))
        # α.shape
        HT = self.__makeH()
        ## convolution like in wavelet domain
        # β = tf.einsum('bijc,uijvco->buvco', α, self.HT) #ok
        β = tf.einsum('bijc,uvijco->buvco', α, HT)
        β = tf.einsum('buvco->buvo',β)
        # β
        y = IDWT2D(self.wave, clean=False)(β)

        # ynatural = tf.einsum('bijc,uijv->buvc', x, tf.cast(self.h_shifted, tf.float32))[0,:,:,0]
        
        return y#, ynatural

    
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.L,
            'wavelet': self.wave,
            'filters': self.filters,
        })
        return config

if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    ## Example
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, N, channels) 
    inputs = tf.keras.Input(shape=input_shape)
    H = LSIVolterra2Dmra(filters=1)
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