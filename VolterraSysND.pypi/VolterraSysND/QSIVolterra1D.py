import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class QSIVolterra1D(tf.keras.layers.Layer):
    def __init__(self, filters=1, kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.filters = filters

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
            name='UIR_kernel'
        )       
        super().build(input_shape)

    def make_h2(self):
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
        # self.h2_shifted = h2_columns_shifted
        # print('h2shifted', self.h2_shifted.shape)
        return h2_columns_shifted


    def call(self, x):
        # x: (batch, N, channels)
        x2 = tf.einsum('bic,bjc->bijc', x, x)  # (batch, N, N, channels)
        # print(x2.shape, self.h2_shifted.shape)
        
        h2_shifted = self.make_h2()
        y = tf.einsum('ijkco, bjkc->bio', h2_shifted, x2)
        return y
        
        

    def get_filter(self):
        return self.h2

        
if __name__ =='__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
    ## Example
    # Functional model
    N, channels, filters = 8, 1, 1
    input_shape = (N, channels)  # Replace N with the actual size of x            #1D
    inputs = tf.keras.Input(shape=input_shape)

    # Apply the custom layer to the inputs
    H = QSIVolterra1D(filters=filters)
    outputs = H(inputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()

    # Random data
    ## 1D
    inputs_data = tf.random.normal((1, N, 1))
    targets = tf.random.normal((1, N, 1))

    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(inputs_data, targets, epochs=5, verbose=1)
