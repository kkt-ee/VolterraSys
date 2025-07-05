""" MEDCNN: Multiresolution Encoder-Decoder Convolutional Neural Network
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
"""


#%%
import tensorflow as tf
# from ConvNDv0 import ConvND

class NLSI2DVolterra(tf.keras.layers.Layer):
    """2D input-output NLSI system with 2nd order Volterra approximation 
    
    Limitation: Be careful with the # of filters

    --kkt@06-06-2025"""
    def __init__(
        self, 
        filters=1, 
        kernel_size=3,
        **kwargs):
        super(NLSI2DVolterra, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
     

    def build(self, input_shape):
        self.inchannels = input_shape[-1]
        # self.filters = input_shape[-1]

        ## h0:= 0th order term
        # Initialize the bias (a0) for the numerator polynomial
        self.a0 = self.add_weight(shape=(self.filters,), initializer='zeros', trainable=True)#, name='h0_bias')

        ## h:= UIR kernle       
        # Create a 2D kernel that will be applied to both spatial dimensions
        self.kernel = self.add_weight(
            name='kernel2d',
            shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True
        )
        self.a1 = self.add_weight(shape=(self.filters,), initializer='zeros', trainable=True)#, name='h0_bias')
        self.a2 = self.add_weight(shape=(self.filters,), initializer='zeros', trainable=True)#, name='h0_bias')


    def call(self, x):
        print(x.shape)
        input_size = x.shape[1]

        # outer product of x
        x2 = tf.einsum('bijp,bklp->bijklp',x,x)
        print('+',x2.shape)
        
        # first order coeffs.
        # h1_out = self.conv2d(x)
        h1_out = tf.nn.convolution(x, self.kernel, padding='SAME')
        
        # 2nd order coeffs
        # h2_out = self.conv4d(x2)#, axis=-1))
        h2_out = self.__separable_conv4d(x2)
        print('h1 h2 out', h1_out.shape, h2_out.shape)

        # reduced quadratic terms: iterative block summation
        h2_out = [[
            tf.einsum('bijklc->bc', h2_out[:, 0:n1+1, 0:n2+1, 0:n1+1, 0:n2+1, :])
            for n1 in range(input_size)]
            for n2 in range(input_size)
        ]
        h2_out = tf.stack([tf.stack(inner_list, axis=1) for inner_list in h2_out], axis=2)
        print('h2_out shape update', h2_out.shape)
              
        # system output
        y = self.a0 + self.a1*h1_out + self.a2*h2_out # + h3_out #+ h4_out + h5_out
        return y

    
    # Function: 4D separable convolution using a single 2D kernel
    def __separable_conv4d(self, x):
        # x: shape [B, N1, N2, N3, N4, C]
        # B, N1, N2, N3, N4, C = x.shape
        # Get static shape for dimensions that shouldn't change
        input_shape = x.shape.as_list()
        N1, N2, N3, N4 = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
        
        # Get dynamic batch size
        B = tf.shape(x)[0]

        ## Step 1: Convolve over (N1, N2)
        x1 = tf.reshape(x, [-1, N1, N2, self.inchannels])  # shape: (B*N3*N4, N1, N2, C)
        y1 = tf.nn.convolution(x1, self.kernel, padding='SAME')
        y1 = tf.reshape(y1, [B, N1, N2, N3, N4, self.filters])   # (B, N1, N2, N3, N4, C)
        y1 = tf.transpose(y1, perm=[0,3,4,1,2,5])
    
        
        # print('+y1 ', y1.shape)


        ## Step 2: Convolve over (N3, N4)
        x2 = tf.reshape(y1, [-1, N3, N4, self.filters])  # shape: (B*N1*N2, N3, N4, C)
        y2 = tf.nn.convolution(x2, self.kernel, padding='SAME')
        y2 = tf.reshape(y2, [B, N1, N2, N3, N4, self.filters])    # final shape
        y2 = tf.transpose(y2, perm=[0,3,4,1,2,5])

        return y2
        
    def get_kernel(self):
        """Returns the kernel weights as a numpy array"""
        return self.kernel.numpy()

    def get_config(self):
        config = super(NLSI2DVolterra, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

    # def get_config(self):
    #     config = super(NLSI2DVolterra, self).get_config()
    #     return config



if __name__ =='__main__':

    # Define input shape and build the model for summary
    input_shape = (16, 16, 2)  # Replace N with the actual size of x
    inputs = tf.keras.Input(shape=input_shape)

    # Create an instance of the custom layer
    H = NLSI2DVolterra(filters=8)
    #conv1d_filters=32, conv1d_kernel_size=3, 
    #  conv2d_filters=32, conv2d_kernel_size=3)

    # Apply the custom layer to the inputs
    outputs = H(inputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Print the model summary
    model.summary()