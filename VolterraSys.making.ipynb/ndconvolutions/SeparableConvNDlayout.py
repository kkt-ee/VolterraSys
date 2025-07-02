import tensorflow as tf
# from tensorflow.keras.layers import Layer

class SeparableConvND(tf.keras.layers.Layer):
    """Separable ND convolution using two 0.5ND kernels mother class

    VolterraSys: Multidimensional linear and nonlinear Volterra kernels in natural and multiresolution bases.
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
    
    --kkt@29-06-2025"""    
    def __init__(self, filters, kernel=None, kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters        
        if kernel is None: 
            self.kernel_size = kernel_size
            self.kernel = kernel 
        else: 
            self.kernel = kernel
            self.kernel_size = tf.shape(kernel)[0]
                    
    def build(self, input_shape):
        self.inchannels = input_shape[-1]
        # self.filters = input_shape[-1]
        ndim = (len(input_shape) - 2)//2  # exclude batch and channel dims
        spatial_shape = input_shape[1:-1]  # [N1, N2, N3, ...]
        

        # Experimental pointwise kernel! a nontrainable pointwise convolution with ones to match 
        # input channels to number of output channels (for smooth separable conv with self.kernel)
        # Determine shape for pointwise kernel (1x1x1...x1, in_channels, out_channels)
        pointwise_shape = (1,) * ndim + (self.inchannels, self.filters) 
        # print('ps', pointwise_shape)
        self.pointwise = self.add_weight(
            name='match_inchannels_with_outchannels_number',
            # shape=(1, 1, input_shape[-1], self.filters),
            shape = pointwise_shape,
            initializer='ones',
            trainable=False)
        if self.kernel is None:
            # Create a ND kernel that will be applied to both spatial dimensions
            # Shape for separable spatial kernel: (K, K, ..., K, filters, filters)
            kernel_shape = (self.kernel_size,) * ndim + (self.filters, self.filters)
            self.kernel = self.add_weight(
                name='separable_kernel',
                # shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
                # shape=(self.kernel_size, self.kernel_size, self.filters, self.filters), ## update after pointwise
                shape=kernel_shape,
                initializer='glorot_uniform',
                trainable=True)
    
    def call(self, x):
        pass

    def get_kernel(self):
        """Returns the kernel weights as a numpy array"""
        return self.kernel.numpy()

    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4], self.filters)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    