import tensorflow as tf
# import keras
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from .LSIVolterraNDlayout import LSIVolterraNDlayout

@tf.keras.utils.register_keras_serializable()
class LSIVolterra1D(LSIVolterraNDlayout):
    """ LSI 1D wavelet and natural basis kernel
    
    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
        
    Linear (m=1) shift invariant multiresolution Volterra kernel for sequences
    Input: sequence x[n]
    Output: Linear shift invariant sequence monomial y1[n]

    --@KKT@04Jul2025 """
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, wave=wave, **kwargs)

    def build(self, input_shape):
        if self.mra:
            N = int(input_shape[1])

            def initialize_synthesis(shape, dtype=None):
                idwt_input = IDWT1D(self.wave, clean=False)
                idwt_input.build((None, N, 1))
                return tf.cast(idwt_input.S, dtype or tf.float32)

            self.S_input = self.add_weight(
                name='input_synthesis',
                shape=(N, N),
                initializer=initialize_synthesis,
                trainable=False,
            )

        super().build(input_shape)

    def __makeH(self):
        paddings = [
        [0, self.N - self.L],  # Depth dimension
        [0,0],
        [0,0]
        ]
        h_padded = tf.pad(self.h1, paddings, mode='CONSTANT', constant_values=0)
        # print(h_padded.shape,'>')

        ## Creating x[n-k] for all n
        n = tf.range(self.N)
        self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)

        h1_shifted = tf.gather(h_padded, self.row_indices, axis=0)
        self.h1_shifted = h1_shifted
        # print(hshifted.shape,'>')
        # print(tf.concat([tf.transpose(DWT1D(self.wave)(tf.transpose(DWT1D(self.wave)(hshifted[...,i:i+1]), perm=[1,0,2])),perm=[1,0,2]) for i in range(self.channels)], axis=-1))

        if self.mra==False: return h1_shifted ## natural shifted kernel
        else: ## shifted MRA kernel

            # ### PERFECT but slow (might have a bug for bior wavelets)
            # HT = tf.stack([
            #         tf.concat([
            #             tf.transpose(DWT1D(self.wave, clean=False)(tf.transpose(DWT1D(self.wave, clean=False)(hshifted[...,i:i+1,j]), perm=[1,0,2])),perm=[1,0,2]) for i in range(self.channels)
            #             ], axis=-1) for j in range(self.filters)], axis=-1)
            # # print(HT.shape)
            # return HT
        
            ## UPDATED
            # Vectorized: flatten last two axes for DWT
            dwt1 = DWT1D(self.wave, clean=False)
            tmp_h_for_dwt = tf.reshape(h1_shifted, (self.N, self.N, self.channels * self.filters))
            ## x is dummy here
            # the following line is a bug for bior wavelets
            # x = dwt1(tmp_h_for_dwt)           # (N, dwt_len, in_channels*out_channels)

            ##bug fix----15 sep 2026
            x = tf.einsum('ric,il->rlc', tmp_h_for_dwt, self.S_input)
            #---------------------



            # 3. Transpose to swap axes 0 and 1 ("rows" and "columns")
            xT = tf.transpose(x, perm=[1, 0, 2])  # (N', N, in_channels * out_channels)

            # 4. DWT1D along the original rows (now axis=1)
            x2 = dwt1(xT)  # (N', N'', in_channels * out_channels)

            # 5. Transpose back to original orientation
            x2T = tf.transpose(x2, perm=[1, 0, 2])  # (N'', N', in_channels * out_channels)

            # 6. Reshape back to (N'', N', in_channels, out_channels)
            # HT = tf.reshape(x2T, (self.N, self.N, self.channels, self.channels))
            # test if above line is a bug by disabling above and enabling below (dont delete any line now just comment out)
            HT = tf.reshape(x2T, (self.N, self.N, self.channels, self.filters))
            return HT

    def call(self, x):
        self.x = x
        if self.mra==True: return self._call_with_mra_kernel(x)   
        else: return self._call_in_natural_domain(x)
    
    def _call_with_mra_kernel(self, x):  
        ## H[v,l] 
        α = DWT1D(self.wave, clean=False)(x) 
        # print('α = ', α.shape)
        HT = self.__makeH()
        print('HT and α', HT.shape, α.shape)
        β = tf.einsum('bic,uico->buo', α, HT)
        # β = tf.einsum('buco->buo',β)
        print('β', β.shape)
        y = IDWT1D(self.wave, clean=False)(β)#
        # print('y',tf.squeeze(y).numpy())
        return y#, ynatural

    def _call_in_natural_domain(self, x):
        h1_shifted = self.__makeH() ##dummy!!
        return tf.einsum('bic,rico->bro', x, self.h1_shifted)

    def sanity_check(self):
        return self._call_with_mra_kernel(self.x), self._call_in_natural_domain(self.x)  



if __name__=='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    ## Example 2
    # Functional model
    N, channels, filters = 4, 1, 1
    input_shape = (N, channels)  # Replace N with the actual size of x            #1D
    inputs = tf.keras.Input(shape=input_shape)
    H = LSIVolterra1D(filters=1, wave='haar')
    H = LSIVolterra1D(filters=1, wave=None)

    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    # Random data
    ## 1D
    x = tf.random.normal((1, N, 1))
    y = tf.random.normal((1, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(x, y, epochs=5, verbose=1)
    del x, y, inputs, outputs

    ## Example 1    
    lay = LSIVolterra1D(filters=1, wave='haar')
    lay = LSIVolterra1D(filters=1, wave='db5')
    lay = LSIVolterra1D(filters=1, wave='bior1.3')
    ## sample batch input
    # x = tf.constant([1,2,3,5,5,3,2,1], dtype=tf.float32)
    # x = tf.constant([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1], dtype=tf.float32)
    # _ = tf.expand_dims(tf.expand_dims(x,axis=-1),axis=0)
    x = tf.random.normal((1, 128, 1))
    print("x shape:", x.shape)
    y = lay(x)
    print("y shape:", y.shape)

    # check if the outputs from both methods are the same
    import matplotlib.pyplot as plt
    y, ynat = lay.sanity_check()
    plt.figure(figsize=(9,2))
    plt.plot(tf.squeeze(y).numpy(), 'o')
    plt.plot(tf.squeeze(ynat).numpy(), '+')
