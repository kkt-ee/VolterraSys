import tensorflow as tf

class TraceConv1Dio(tf.keras.layers.Layer):
    """Custom layer for batched circular cross-correlation with channels"""
    
    def __init__(self, **kwargs):
        super(TraceConv1Dio, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # input_shape: (batch_size, N, N, channels)
        self.N = input_shape[1]
        channels = input_shape[-1]
        
        # # Initialize learnable kernel
        # self.h2 = self.add_weight(
        #     name='h2_kernel',
        #     shape=(self.N, self.N, channels),
        #     initializer='glorot_uniform',
        #     trainable=True
        # )
        
        # Precompute circular indices
        n = tf.range(self.N)
        self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)
        
        super(TraceConv1Dio, self).build(input_shape)
    
    def call(self, inputs, h2):
        """
        Args:
            inputs: Tensor of shape (batch_size, N, N, channels)
        Returns:
            Tensor of shape (batch_size, N)
        """
        self.h2 = h2
        # print('h+',self.h2.shape)
        batch_size = tf.shape(inputs)[0]
        # print('r+',self.row_indices.shape)
        # Expand indices for batch dimension
        batch_indices = tf.expand_dims(self.row_indices, 0)  # (1, N, N)
        # print('bi+',batch_indices.shape)
        batch_indices = tf.tile(batch_indices, [batch_size, 1, 1])  # (batch, N, N)
        # print('bi+',batch_indices.shape)

        # Gather shifted rows and columns
        rows_shifted = tf.gather(inputs, self.row_indices, axis=1)  # (batch, N, N, N, ch)
        columns_shifted = tf.gather(
            rows_shifted, 
            batch_indices,  # Use batch-aware indices
            axis=3, 
            batch_dims=2    # Match batch dimension
        )  # (batch, N, N, N, ch)
        # print('rc+',rows_shifted.shape, columns_shifted.shape, self.h2.shape)

        # Compute cross-correlation with learnable kernel
        result = tf.einsum('bnijc,ijc->bnc', columns_shifted, self.h2)
        return result
    
    def get_config(self):
        base_config = super(TraceConv1Dio, self).get_config()
        return base_config