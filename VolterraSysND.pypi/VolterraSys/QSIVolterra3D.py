import tensorflow as tf
# from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWT3DFB import DWT3D, IDWT3D
from VolterraSys.QSIVolterraNDlayout import QSIVolterraNDlayout

@tf.keras.utils.register_keras_serializable()
class QSIVolterra3D(QSIVolterraNDlayout):
    """ QSI Volterra 3D I/O in wavelet and in natural domain

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
    --kkt@3Jul2025"""
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, wave=wave, **kwargs)
        # Add any other initialization you need here 
    
    def _make_h2_shifted_mra_kernel3(self):
        self.h2 = self._make_h2()
        h2_shifted = self.make_h2_shifted_kernel()      
          
        print('h2_shifted', h2_shifted.shape)
        shape = tf.shape(h2_shifted) # ijk pqr stu co, for example (4, 4, 4, 4, 4, 4, 1, 1)
        _ = tf.reshape(h2_shifted, [shape[0]*shape[1]*shape[2]*shape[3]*shape[4]*shape[5], shape[6], shape[7] , shape[8], shape[9]*shape[10]])
        _ = DWT3D(self.wave, clean=False)(_) #stu
        _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7], shape[8], shape[9],shape[10]])
        _ = tf.reshape(_,  [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5], shape[6]*shape[7]*shape[8]*shape[9]*shape[10]])
        _ = DWT3D(self.wave, clean=False)(_) #pqr
        _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7], shape[8], shape[9],shape[10]])
        _ = tf.transpose(_, perm=[3,4,5, 0,1,2, 6,7,8,9,10])
        _ = tf.reshape(_, [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5], shape[6]*shape[7]*shape[8]*shape[9]*shape[10]])
        _ = DWT3D(self.wave, clean=False)(_) #ijk
        _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7], shape[8], shape[9],shape[10]])
        H = tf.transpose(_, perm=[3,4,5, 0,1,2, 6,7,8,9,10])
        print('H',H.shape)
        return H

    def call(self, x):
        self.x = x
        if self.mra==True: return self._call_compute_with_mra_kernel3(x)   
        else: return self._call_compute_in_natural_domain(x)

    def _call_compute_with_mra_kernel3(self, x):
        # x: (batch, N, channels)
        # x2 = tf.einsum('bijkc,bpqrc->bijkpqrc', x, x)  # (batch, N, N, channels)
        # print('x2', x2.shape)
        x2_eq, y2_eq = self.get_einsum_strings(self.dim)
        x2 = tf.einsum(x2_eq, x, x) 
        def dwt6(x2): 
            shape = tf.shape(x2)  ## b ijk pqr c
            _ = tf.reshape(x2, [shape[0]*shape[1]*shape[2]*shape[3], shape[4], shape[5], shape[6], shape[7]])
            # print('_',_.shape)
            _ = DWT3D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            _ = tf.transpose(_, perm=[0, 4,5,6, 1,2,3, 7])
            _ = tf.reshape(_, [shape[0]*shape[1]*shape[2]*shape[3], shape[4], shape[5], shape[6], shape[7]])
            _ = DWT3D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            return tf.transpose(_, perm=[0, 4,5,6, 1,2,3, 7])
        α2 = dwt6(x2)
        print('α2', α2.shape)
        H = self._make_h2_shifted_mra_kernel3()
        ## Fitlering
        β = tf.einsum('ijkpqrstuco,bpqrstuc->bijko', H, α2)
        print('β', β.shape)

        # print(x2.shape, self.h2_shifted.shape)
        # y = tf.einsum('ijkco, bjkc->bio', self.h2_shifted, x2)
        # y2 = self.idwt1(β)
        y2 = IDWT3D(self.wave, clean=False)(β)
        print('y2', y2.shape)
        return y2    

    def sanity_check(self):
        return self._call_compute_with_mra_kernel3(self.x), self._call_compute_in_natural_domain(self.x)  

        
if __name__ =='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    ## Example
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, N, N, channels)  # Replace N with the actual size of x       #2D
    inputs = tf.keras.Input(shape=input_shape)
    # Apply the custom layer to the inputs
    # H = QSIVolterra2D(filters=filters, wave=None)
    H = QSIVolterra3D(filters=filters, wave='db2')
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 2D Random data
    x = tf.random.normal((1, N, N, N, 1))
    y = tf.random.normal((1, N, N, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(x, y, epochs=5, verbose=1)    