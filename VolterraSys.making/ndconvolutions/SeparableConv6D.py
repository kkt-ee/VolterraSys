# import tensorflow as tf
# from tensorflow.keras.layers import Layer

# # # include ../dirx 
# mylibpath = [
#     '/data1/kishoretarafdar/src.port/VolterraMRAsystems.v0/VolterraSys/ndconvolutions'
#     ]
# import sys
# [sys.path.insert(0,_) for _ in mylibpath]
# del mylibpath


import tensorflow as tf
from SeparableConvNDlayout import SeparableConvND


class SeparableConv6D(SeparableConvND):
    """Separable 6D convolution using 3D kernels

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
        super().__init__(filters=filters, kernel=kernel, kernel_size=kernel_size, **kwargs)
        
    def call(self, inputs):
        return self.__separable_conv6d(inputs)


    def __separable_conv6d(self, x):
        # x: [B, N1, N2, N3, N4, N5, N6, C]
        input_shape = x.shape.as_list()
        N1, N2, N3, N4, N5, N6 = input_shape[1], input_shape[2], input_shape[3], input_shape[4], input_shape[5], input_shape[6]
        B = tf.shape(x)[0]

        # O = kernel3d.shape[-1]
        # h2 = kernel3d

        # Step 1: Convolve over (N1, N2, N3)
        x1 = tf.reshape(x, [-1, N4, N5, N6, self.inchannels])  # shape: (B*N3*N4, N1, N2, C)
        # x1 = tf.reshape(x, [B * N4 * N5 * N6, N1, N2, N3, self.inchannels])
        # print('x', x.shape)
        # print(x1.shape)
        x1 = tf.nn.convolution(x1, self.pointwise, padding='SAME')
        y1 = tf.nn.convolution(x1, self.kernel, padding='SAME')
        print('kernel', self.kernel.shape)
        # print(y1.shape)
        y1 = tf.reshape(y1, [B, N1, N2, N3, N4, N5, N6, self.filters])
        # print(y1.shape)
        y1 = tf.transpose(y1, perm=[0,4,5,6,1,2,3,7])
    
        # Step 2: Convolve over (N4, N5, N6)
        x2 = tf.reshape(y1, [-1, N1, N2, N3, self.filters])  # shape: (B*N1*N2, N3, N4, C) ## OK
        # x2 = tf.reshape(y1, [B * N1 * N2 * N3, N4, N5, N6, self.inchannels])
        # print(x2.shape)
        y2 = tf.nn.convolution(x2, self.kernel, padding='SAME')
        # print(y2.shape)
        y2 = tf.reshape(y2, [B, N4, N5, N6, N1, N2, N3, self.filters])
        # print(y2.shape)
        y2 = tf.transpose(y2, perm=[0,4,5,6,1,2,3,7])
        return y2
        

if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

    # Create a model using the layer
    inputs = tf.keras.Input(shape=(16, 16, 16, 16, 16, 16, 2))  # (N1, N2, N3, N4, C)
    # inputs = tf.keras.Input(shape=(16, 16, 16, 8, 8, 8, 2))  # (N1, N2, N3, N4, C)
    outputs = SeparableConv6D(filters=7, kernel_size=3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # # Test with random data
    test_input = tf.random.normal([2, 16, 16, 16, 16, 16, 16, 2])
    output = model(test_input)
    print(output.shape)
    del outputs, model, inputs, test_input, output



    ## example 2
    # Input dimensions
    B, N1, N2, N3, N4, N5, N6, C = 2, 8, 8, 8, 8, 8, 8, 3  # Batch size, 4D volume size, channels
    # B, N1, N2, N3, N4, C = 1, 2, 2, 2, 2, 1  # Batch size, 4D volume size, channels
    # B, N1, N2, N3, N4, C = 1, 3, 3, 2, 2, 1  # Batch size, 4D volume size, channels
    x = tf.random.normal((B, N1, N2, N3, N4, N5, N6, C))
    # x.shape
    layer = SeparableConv6D(filters=5, kernel_size=3)
    yout = layer(x)
    kernel3d = layer.get_kernel()
    print("Kernel shape:", kernel3d.shape)  # Should be (3, 3, 2, 32)
    print('yout', yout.shape)
