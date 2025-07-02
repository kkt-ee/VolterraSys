
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


class SeparableConv4D(SeparableConvND):
    """Separable 4D convolution using 2D kernels

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
        return self.__separable_conv4d(inputs)
        

    # Function: 4D separable convolution using a single 2D kernel
    def __separable_conv4d(self, x):
        # x: shape [B, N1, N2, N3, N4, C]
        # B, N1, N2, N3, N4, C = x.shape
        # Get static shape for dimensions that shouldn't change
        input_shape = x.shape.as_list()
        N1, N2, N3, N4 = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        
        # Get dynamic batch size
        B = tf.shape(x)[0]

        # print(' x ', x.shape)
        ## Step 1: Convolve over (N3, N4)
        x1 = tf.reshape(x, [-1, N3, N4, self.inchannels])  # shape: (B*N3*N4, N1, N2, C)
        x1 = tf.nn.convolution(x1, self.pointwise, padding='SAME')
        y1 = tf.nn.convolution(x1, self.kernel, padding='SAME')
        # print(f'+pointwise kernel {self.pointwise.shape} \n+kernel {self.kernel.shape}, \n x1 {x1.shape}, \n y1 { y1.shape}')
        # y1 = tf.reshape(y1, [B, N1, N2, N3, N4, self.inchannels])   # (B, N1, N2, N3, N4, C) ## OK
        y1 = tf.reshape(y1, [B, N1, N2, N3, N4, self.filters])   # (B, N1, N2, N3, N4, C) 
        # print(' y1 ', y1.shape)
        y1 = tf.transpose(y1, perm=[0,3,4,1,2,5])           
        # print('Ty1 ', y1.shape)

        ## Step 2: Convolve over (N1, N2)
        # x2 = tf.reshape(y1, [-1, N1, N2, self.inchannels])  # shape: (B*N1*N2, N3, N4, C) ## OK
        x2 = tf.reshape(y1, [-1, N1, N2, self.filters])  # shape: (B*N1*N2, N3, N4, C) ## OK
        # print(' x2 or reshape y1 ', x2.shape)
        y2 = tf.nn.convolution(x2, self.kernel, padding='SAME')
        # print(' y2 ', y2.shape)
        # y2 = tf.reshape(y2, [B, N1, N2, N3, N4, self.inchannels])    # final shape ## OK
        y2 = tf.reshape(y2, [B, N3, N4, N1, N2,  self.filters])    # final shape
        # print('+y2 ', y2.shape)
        
        y2 = tf.transpose(y2, perm=[0,3,4,1,2,5])
        # print('Ty2 ', y2.shape)
        return y2
        
if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    
    # Create a model using the layer
    inputs = tf.keras.Input(shape=(16, 16, 16, 16, 2))  # (N1, N2, N3, N4, C)
    inputs = tf.keras.Input(shape=(128, 128, 64, 64, 2))  # (N1, N2, N3, N4, C)
    inputs = tf.keras.Input(shape=(64, 64, 128, 128, 2))  # (N1, N2, N3, N4, C)
    outputs = SeparableConv4D(filters=7, kernel_size=3)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # model.build(None)
    model.summary()

    # # # Test with random data
    # test_input = tf.random.normal([2, 16, 16, 16, 16, 2])
    # output = model(test_input)
    # print(output.shape)
    # del outputs, model, inputs, test_input, output


    ## Example2
    # Input dimensions
    B, N1, N2, N3, N4, C = 2, 8, 8, 8, 8, 3  # Batch size, 4D volume size, channels
    # B, N1, N2, N3, N4, C = 1, 2, 2, 2, 2, 1  # Batch size, 4D volume size, channels
    # B, N1, N2, N3, N4, C = 1, 3, 3, 2, 2, 1  # Batch size, 4D volume size, channels
    x = tf.random.normal((B, N1, N2, N3, N4, C))
    x.shape

    # layer = SeparableConv4D(filters=x.shape[-1], kernel_size=3) ## OK
    layer = SeparableConv4D(filters=13, kernel_size=3)
    yout = layer(x)
    kernel2d = layer.get_kernel()
    print("Kernel shape:", kernel2d.shape)  # Should be (3, 3, 2, 32)
    yout.shape
