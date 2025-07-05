
#%% Shift invariant Linear (m=1) Multiresolution Volterra Kernel
import tensorflow as tf
import keras
from TFDWT.DWT1DFB import DWT1D, IDWT1D

@tf.keras.utils.register_keras_serializable()
class LSIVolterraNDlayout(tf.keras.layers.Layer):
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

    --@KKT@04Jul2025"""
    
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
            'wavelet': self.wave,
            'filters': self.filters,
            'mra': self.mra
        })
        return config