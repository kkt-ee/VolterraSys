import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWT2DFB import DWT2D, IDWT2D

@tf.keras.utils.register_keras_serializable()
class QSIVolterra1Dmra(tf.keras.layers.Layer):
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.filters = filters
        self.wave = wave
        # self.dwt2 = DWT2D(self.wave, clean=False)
        # self.idwt1 = IDWT1D(self.wave, clean=False)

    def build(self, input_shape):
        self.N = input_shape[1]         # sequence length
        self.channels = input_shape[-1] # number of channels
        # Multi-channel kernel: (L, channels)
        kernel_shape = (self.L, self.channels, self.filters)
        self.h = self.add_weight(
            # shape=(self.L, self.channels),
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            name='UIR_kernel')
        super().build(input_shape)
    
    def make_mra_h2(self):
        # Quadratic kernel outer product: (L, channels) x (L, channels) -> (L, L, channels)
        h2 = tf.einsum('ico,jco->ijco', self.h, self.h)  # (L, L, channels)
        self.h2 = h2
        # Pad to (N, N, channels)
        paddings = [
            [0, self.N - self.L],
            [0, self.N - self.L],
            [0, 0],
            [0, 0]
        ]
        h2_padded = tf.pad(h2, paddings, mode='CONSTANT', constant_values=0)
        # self.h2_padded = h2_padded
        ## creating h[n-k1,n-k2]
        # N = Npoint
        n = tf.range(self.N)
        # Create circular indices for all possible shifts
        # row_indices = tf.math.mod(n[:, tf.newaxis] - tf.range(N), L)  # [N, N]
        row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)
        h2_rows_shifted = tf.gather(h2_padded, row_indices, axis=0)         # [N, N, N]
        h2_columns_shifted = tf.gather(h2_rows_shifted, row_indices,   # [N, N, N]
                                        axis=2, batch_dims=1)
        h2_shifted = h2_columns_shifted
        # print('h2shifted', self.h2_shifted.shape)

        def make_mra_kernel(h_shifted):
            shape = tf.shape(h_shifted)
            _ = tf.reshape(h_shifted, [shape[0], shape[1], shape[2], shape[3] * shape[4]])
            _ = DWT2D(self.wave, clean=False)(_)
            h_init = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4]])
            # print(h_init.shape, tf.squeeze(h_init))

            # dwt of third axis
            shape = tf.shape(h_init)
            shape
            _ = tf.reshape(h_init, [shape[0], shape[1], shape[2]*shape[3]*shape[4]])
            _ = tf.transpose(_, perm=[1,0,2])
            _ = DWT1D(self.wave, clean=False)(_)
            _ = tf.transpose(_, perm=[1,0,2])
            H = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4]])
            return H
        H = make_mra_kernel(h2_shifted)
        print('H',H.shape)
        return H

        # ## update this for batched multichannel
        # _ = tf.expand_dims(tf.cast(h2_shifted, dtype=tf.float32), axis=-1)
        # h_init = DWT2D(self.wave, clean=False)(_)
        # print(tf.squeeze(h_init))

        # H = tf.expand_dims(
        #     tf.transpose(
        #         DWT1D(wave, clean=False)(tf.transpose(tf.squeeze(h_init, axis=-1), perm=[1,0,2])),
        #         perm=[1,0,2]),axis=-1)
        # H[:,:,:,0]

    def call(self, x):
        # x: (batch, N, channels)
        x2 = tf.einsum('bic,bjc->bijc', x, x)  # (batch, N, N, channels)
        print('x2', x2.shape)
        α2 = DWT2D(self.wave, clean=False)(x2)
        print('α2', α2.shape)
        H = self.make_mra_h2()
        ## Fitlering
        β = tf.einsum('ijkco,bjkc->bio', H, α2)
        print('β', β.shape)

        # print(x2.shape, self.h2_shifted.shape)
        # y = tf.einsum('ijkco, bjkc->bio', self.h2_shifted, x2)
        # y2 = self.idwt1(β)
        y2 = IDWT1D(self.wave, clean=False)(β)
        print('y2', y2.shape)
        return y2

    def get_filter(self):
        return self.h2

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.L,
            'filters': self.filters,
            'wavelet': self.wave
        })
        return config

        
if __name__ =='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

    # Define input shape and build the model for summary
    N = 128
    input_shape = (N, 1)  # Replace N with the actual size of x
    inputs = tf.keras.Input(shape=input_shape)

    # Create an instance of the custom layer
    # h2 = np.array([[1, 2], 
    #           [1, 3]]).astype(np.float32)
    # H = TraceConv1Dio(kernel=tf.expand_dims(h2, axis=-1))
    # (tf.expand_dims(tf.expand_dims(x2,axis=0),axis=-1))

    #conv1d_filters=32, conv1d_kernel_size=3, 
    #  conv2d_filters=32, conv2d_kernel_size=3)

    # Apply the custom layer to the inputs
    H = QSIVolterra1Dmra()
    outputs = H(inputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Print the model summary
    model.summary()
