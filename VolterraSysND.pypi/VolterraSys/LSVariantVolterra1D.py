import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D

#%% Shift invariant Linear (m=1) Multiresolution Volterra Kernel
class LSVariantVolterra1D(tf.keras.layers.Layer):
    """
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
        
    
        Shift variant Linear (m=1) Multiresolution Volterra Kernel
        Input: sequence x[n]
        Output: Shift variant linear monomial y1, i.e., m=1

    --@KKT, 03Jul2025"""
    
    def __init__(self, filters=1, Ny=16, wave='haar', mra=True, **kwargs):
        super().__init__(**kwargs)
        # self.L = kernel_size
        self.wave = wave
        self.filters = filters
        self.Ny = Ny
        self.mra = mra
    
    def build(self, input_shape):
        # input_shape: (batch_size, N, N, channels)
        self.N = input_shape[1]
        # self.L = self.N
        self.channels = input_shape[-1]
        kernel_shape = (self.Ny, self.N, input_shape[-1], self.filters)
        # kernel_shape = (self.L, input_shape[-1])
        self.h1 = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            # regularizer=self.kernel_regularizer,
            name='kernel'
        )
        # n = tf.range(self.N)
        # self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)
        super().build(input_shape)

    def call(self, x):  
        self.x = x  
        if self.mra==True:
            return self.__compute_output_with_mra_kernel(x)
        else: 
            return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        HT = self.__make_H()
        # print(HT.shape,'HT')
        ## l-axis below --->l and ^v
        
        ## H[v,l] 
        α = DWT1D(self.wave, clean=False)(x)
        # print('α = ', α.shape)
        # print('HT and α', HT.shape, α.shape)
        β = tf.einsum('bic,uico->buo',α, HT) #HT and α (16, 4, 1, 1) (None, 4, 1)
        # β = tf.einsum('buco->buo',β)
        # print('β', β.shape)

        # HTα = HT * α
        # print(HTα.shape, '\n', tf.squeeze(HTα).numpy())
        # beta = tf.expand_dims(tf.einsum('ijk->ik',HTα), axis=-1)
        
        # βT = tf.transpose(beta, perm=[1,0,2])
        # print('βT',tf.squeeze(βT).numpy())
        y = IDWT1D(self.wave, clean=False)(β)#
        # print('y',tf.squeeze(y).numpy())
        return y
         
    def __compute_output_in_natural_domain(self, x):
        ynatural = tf.einsum('bic,uico->buo',x, self.h1)
        # print(ynatural.shape, 'natural y') 
        return ynatural

    def sanity_check(self):
        return self.__compute_output_with_mra_kernel(self.x),  self.__compute_output_in_natural_domain(self.x)  
    
    def __make_H(self):
        def mra_kernel(h):
            shape = tf.shape(h)
            Ny, N, channels, filters = shape[0], shape[1], shape[2], shape[3]
            
            # Combine channels and filters: shape becomes (Ny, N, in_channels * filters)
            h1_reshaped = tf.reshape(self.h1, [Ny, N, -1])
            # print(h1_reshaped.shape, 'h1_reshaped')

            # Apply DWT1D twice
            dwt_layer1 = DWT1D(self.wave, clean=False)
            dwt_layer2 = DWT1D(self.wave, clean=False)
            
            dwt_once = dwt_layer1(h1_reshaped)
            # print(dwt_once.shape, 'dwt once')
            dwt_once_T = tf.transpose(dwt_once, perm=[1, 0, 2])
            # print(dwt_once_T.shape, 'dwt once T')  
            dwt_twice = dwt_layer2(dwt_once_T)
            # print(dwt_twice.shape, 'dwt twice')

            # Final transpose to (Ny, N, in_channels * filters)
            tmp_H = tf.transpose(dwt_twice, perm=[1, 0, 2])

            # Reshape back to (Ny, N, in_channels, filters)
            H = tf.reshape(tmp_H, [self.h1.shape[0], self.h1.shape[1], self.h1.shape[2], self.h1.shape[3]])
            # print(H.shape,'in mra kernel HT')
            return H
        
        if self.mra == True:
            return mra_kernel(self.h1)
        else:
            return self.h1
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'Ny': self.Ny,
            'wavelet': self.wave,
            'filters': self.filters,
            'mra': self.mra
        })
        return config



if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
    ## Example
    # Functional model
    N, channels = 4, 1
    input_shape = (N, channels) 
    inputs = tf.keras.Input(shape=input_shape)
    filters, Ny = 1, 16
    H = LSVariantVolterra1D(filters=filters, Ny=Ny, wave='haar')
    # H = LSVariantVolterra1D(filters=filters, Ny=Ny, wave='haar', mra=False)
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 1D
    inputs_data = tf.random.normal((1, N, 1))
    targets = tf.random.normal((1, Ny, 1))
    # Training loop for 5 epochs
    epochs=5
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
