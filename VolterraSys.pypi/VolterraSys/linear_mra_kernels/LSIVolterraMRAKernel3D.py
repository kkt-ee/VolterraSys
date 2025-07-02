import tensorflow as tf
import keras
from TFDWT.DWT3DFB import DWT3D, IDWT3D

@tf.keras.utils.register_keras_serializable()
class LSIVolterraMRAKernel3D(tf.keras.layers.Layer):
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
       Input: volumetric image x[n1,n2,n3]
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

        kernel_shape = (self.L, self.L, self.L, input_shape[-1], self.filters)
        # kernel_shape = (self.L, self.L, input_shape[-1])
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            # regularizer=self.kernel_regularizer,
            name='kernel')
        super().build(input_shape)
        
    def __makeH(self):
        paddings = [
            [0, self.N - self.L],  # depth dimension
            [0, self.N - self.L],  # height dimension
            [0, self.N - self.L],  # width dimension
            [0, 0],
            [0, 0] 
        ]
        h1 = tf.pad(self.h1, paddings, mode='CONSTANT', constant_values=0)

        # Create a meshgrid for n1, n2, n3, k1, k2, k3
        n1 = tf.range(self.N)  # depth indices
        n2 = tf.range(self.N)  # height indices
        n3 = tf.range(self.N)  # width indices
        k1 = tf.range(self.N)  # depth shift
        k2 = tf.range(self.N)  # height shift
        k3 = tf.range(self.N)  # width shift

        # meshgrid for all dims
        n1, n2, n3, k1, k2, k3 = tf.meshgrid(n1, n2, n3, k1, k2, k3, indexing='ij')  # Shape (N, N, N, N, N, N)

        # Compute circularly shifted indices
        shifted_n1 = tf.math.mod(n1 - k1, self.N)  # Depth shift
        shifted_n2 = tf.math.mod(n2 - k2, self.N)  # Height shift
        shifted_n3 = tf.math.mod(n3 - k3, self.N)  # Width shift

        # Gather the values from h1 using the shifted indices
        # h1: shape [N, N, N, ...]
        # Need to stack the indices as [shifted_n1, shifted_n2, shifted_n3]
        idx = tf.stack([shifted_n1, shifted_n2, shifted_n3], axis=-1)  # shape (..., 3)
        h1_shifted = tf.gather_nd(h1, idx)  # If h1 has more dimensions, ND gathers all trailing dims

        # h1_shifted = tf.transpose(h1_shifted,perm=[0,2,3,1,4,5])  # n1,:,:,n2 01234 ## very important line!!
        print('++', h1_shifted.shape)

        
        def dwtmra(h1_shifted):
            # ### second vectorized upgrade!! not matching
            # # h1_shifted: (N, N, N, N, in_channels, out_channels)
            N, in_channels, out_channels = self.N, self.channels, self.filters
            num_io = in_channels * out_channels

            # Flatten (in, out) for DWT2D
            hflat = tf.reshape(h1_shifted, (N, N, N, N, N, N, num_io))
            hflat_reshape = tf.reshape(hflat, (N*N*N, N, N, N, num_io))

            dwt3d = DWT3D(self.wave, clean=False)
            x1 = dwt3d(tf.cast(hflat_reshape, tf.float32))
            x1r = tf.reshape(x1, (N, N, N, N, N, N, num_io))

            # Transpose for columnplanewise DWT2D
            x1T = tf.transpose(x1r, [3, 4, 5, 0, 1, 2, 6])     # <--- This is the key fix
            x1T_reshape = tf.reshape(x1T, (N*N*N, N, N, N, num_io))
            x2 = dwt3d(x1T_reshape)
            x2r = tf.reshape(x2, (N, N, N, N, N, N, num_io))

            # Transpose back
            HT = tf.transpose(x2r, [3, 4, 5, 0, 1, 2, 6])
            HT = tf.reshape(HT, (N, N, N, N, N, N, in_channels, out_channels))
            return HT
        
        return dwtmra(h1_shifted)    

    def call(self, x):

        # HT = self.__getH()

        α = DWT3D(self.wave, clean=False)(tf.cast(x, dtype=tf.float32))
        # α.shape

        H = self.__makeH()
        ## convolution like in wavelet domain
        β = tf.einsum('bijkc,uvwijkco->buvwco', α, H)
        β = tf.einsum('buvwco->buvwo',β)
        # β
        y = IDWT3D(self.wave, clean=False)(β)

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
    input_shape = (N, N, N, channels)  # Replace N with the actual size of x    #3D
    inputs = tf.keras.Input(shape=input_shape)
    # Apply the custom layer to the inputs
    H = LSIVolterraMRAKernel3D(filters=1)
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 3D Random data
    inputs_data = tf.random.normal((1, N, N, N, 1))
    targets = tf.random.normal((1, N, N, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
    