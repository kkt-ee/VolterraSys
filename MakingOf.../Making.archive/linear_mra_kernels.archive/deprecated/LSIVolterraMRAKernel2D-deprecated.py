import tensorflow as tf
import keras
from TFDWT.DWT2DFB import DWT2D, IDWT2D

@tf.keras.utils.register_keras_serializable()
class LSIVolterraMRAKernel2D(tf.keras.layers.Layer):
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

    --@KKT@12Feb2025"""
    
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
            name='kernel'
        )

        def __makeH():
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
            h1_shifted = tf.transpose(h1_shifted,perm=[0,2,3,1,4,5])  # n1,:,:,n2 01234
            print('++', h1_shifted.shape)

            def foo(h1_shifted):
                # Example placeholder tensor
                # h1_shifted shape: (N, N, N, N, in_channels, out_channels)
                # You can replace these with actual values
                N = self.N
                in_channels = self.channels
                out_channels = self.filters
                # h1_shifted = tf.random.normal((N, N, N, N, in_channels, out_channels))

                def reshape1(h1_shifted):
                    ##### Step 1: Transpose to bring in_channels and out_channels to the front
                    # New shape: (in_channels, out_channels, N, N, N, N)
                    h1_transposed = tf.transpose(h1_shifted, perm=[4, 5, 0, 1, 2, 3])

                    # Step 2: Reshape to (out_channels * in_channels * N, N, N, N)
                    new_shape = (out_channels * in_channels * N, N, N, N)
                    h1_reshaped = tf.reshape(h1_transposed, new_shape)
                    print(h1_reshaped.shape)
                    return h1_reshaped
        
                h1_reshaped = reshape1(h1_shifted)
                transform1 = DWT2D(self.wave, clean=False)(tf.cast(h1_reshaped, dtype=tf.float32))

                def reshape1back(transform1):
                    #####get back to original shape
                    # Step 1: Reshape back to (in_channels, out_channels, N, N, N, N)
                    transform1_back = tf.reshape(transform1, (in_channels, out_channels, N, N, N, N))

                    # Step 2: Transpose back to original shape: (N, N, N, N, in_channels, out_channels)
                    transform1_back = tf.transpose(transform1_back, perm=[2, 3, 4, 5, 0, 1])
                    print(transform1_back.shape)
                    return transform1_back

                transform1_back = reshape1back(transform1)
                transform1_back = tf.transpose(transform1_back, perm=[1,0,3,2,4,5])

                transform1_back_reshaped = reshape1(transform1_back)
                transform2 = DWT2D(self.wave, clean=False)(tf.cast(transform1_back_reshaped, dtype=tf.float32))
                transform2_back = reshape1back(transform2)
                transform2_back = tf.transpose(transform2_back, perm=[1,0,3,2,4,5])
                return transform2_back

            h1_n_l = foo(h1_shifted)
            HT = h1_n_l
            return HT

        self.HT = __makeH()
        super().build(input_shape)
    

    def call(self, x):
        α = DWT2D(self.wave, clean=False)(tf.cast(x, dtype=tf.float32))
        # α.shape

        ## convolution like in wavelet domain
        β = tf.einsum('bijc,uijvco->buvco', α, self.HT)
        β = tf.einsum('buvco->buvo',β)
        # β
        y = IDWT2D(self.wave, clean=False)(β)        
        return y

    
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.L,
            'wavelet': self.wave,
            'filters': self.filters,
        })
        return config

if __name__=='__main__':
    lay = LSIVolterraMRAKernel2D(filters=1)
    y = lay(xx)
    y.shape