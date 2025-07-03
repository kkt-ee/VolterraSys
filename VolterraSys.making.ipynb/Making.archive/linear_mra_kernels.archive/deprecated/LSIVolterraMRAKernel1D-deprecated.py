
#%% Shift invariant Linear (m=1) Multiresolution Volterra Kernel
import tensorflow as tf
import keras
from TFDWT.DWT1DFB import DWT1D, IDWT1D

@tf.keras.utils.register_keras_serializable()
class LSIVolterraMRAKernel1D(tf.keras.layers.Layer):
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
        Input: sequence x[n]
        Output: Shift invariant linear monomial y1, i.e., m=1

    --KKT@21Jun2025"""
    
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.wave = wave
        self.filters = filters
    
    def build(self, input_shape):
        # input_shape: (batch_size, N, N, channels)
        self.N = input_shape[1]
        self.channels = input_shape[-1]
   
        kernel_shape = (self.L, input_shape[-1], self.filters)
        # kernel_shape = (self.L, input_shape[-1])
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            # regularizer=self.kernel_regularizer,
            name='kernel'
        )
        n = tf.range(self.N)
        self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)

        def __makeH():
            paddings = [
            [0, self.N - self.L],  # Depth dimension
            [0,0],
            [0,0]
            ]
            h_padded = tf.pad(self.h1, paddings, mode='CONSTANT', constant_values=0)
            # print(h_padded.shape,'>')

            ## Creating x[n-k] for all n
            hshifted = tf.gather(h_padded, self.row_indices, axis=0)
            HT = tf.stack([
                    tf.concat([
                        tf.transpose(DWT1D(self.wave, clean=False)(tf.transpose(DWT1D(self.wave, clean=False)(hshifted[...,i:i+1,j]), perm=[1,0,2])),perm=[1,0,2]) for i in range(self.channels)
                        ], axis=-1) for j in range(self.filters)], axis=-1)
            # print(HT.shape)
            return HT
        self.HT = __makeH()
        super().build(input_shape)
    
    def call(self, x):
        α = DWT1D(self.wave, clean=False)(x) 
        # print('α = ', α.shape)
        # print('HT and α', self.HT.shape, α.shape)
        β = tf.einsum('bic,uico->buco', α, self.HT)
        β = tf.einsum('buco->buo',β)
        y = IDWT1D(self.wave, clean=False)(β)
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
    lay = LSIVolterraMRAKernel1D(filters=1)
    # y = lay(xx)
    # y = lay(tf.concat([xx,xx], axis=-1))
    # y.shape