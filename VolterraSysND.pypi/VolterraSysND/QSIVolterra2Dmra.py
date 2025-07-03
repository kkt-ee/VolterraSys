import tensorflow as tf
from TFDWT.DWT1DFB import DWT1D, IDWT1D
from TFDWT.DWT2DFB import DWT2D, IDWT2D

@tf.keras.utils.register_keras_serializable()
class QSIVolterra2Dmra(tf.keras.layers.Layer):
    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.filters = filters
        self.wave = wave
        # self.dwt2 = DWT2D(self.wave, clean=False)
        # self.idwt1 = IDWT1D(self.wave, clean=False)

    def build(self, input_shape):
        self.N = input_shape[1]         # sequence length
        self.axes = list(range(1, len(input_shape) - 1))
        self.dim = self.axes[-1]
        self.channels = input_shape[-1] # number of channels
        # Multi-channel kernel: (L, channels)
        # kernel_shape = (self.L, self.channels, self.filters)
        kernel_shape = [self.L] * self.dim + [self.channels, self.filters]
        self.h = self.add_weight(
            # shape=(self.L, self.channels),
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            name='UIR_kernel')
        super().build(input_shape)
        
    
    def make_h2(self):
        ## make h2: ## Quadratic kernel outer product: (L, channels) x (L, channels) -> (L, L, channels)
        # if self.dim==1:
        #     h2 = tf.einsum('ico,jco->ijco', h, h)  # (L, L, channels)       
        if self.dim==2:
            h2 = tf.einsum('ijco,klco->ijklco', self.h, self.h)  # (L, L, channels)
        # elif self.dim==3:
        #     h2 = tf.einsum('ijkco,pqrco->ijkpqrco', h, h)  # (L, L, channels)
        else:
            raise NotImplementedError("Hardcoded einsum strings supported for dim=1,2")
        self.h2 = h2
  
        def tf_pad_nd_volterra(h2, N, L, D):
            """
            Pad an ND Volterra kernel tensor from shape [L]*2D + [c, f] to [N]*2D + [c, f], using only TensorFlow ops.

            Args:
                h2: tf.Tensor, shape [L, ... L] (2D times), [c, f]
                N: int, target spatial size
                L: int, kernel spatial size
                D: int, number of spatial dimensions

            Returns:
                h2_padded: tf.Tensor, shape [N]*2D + [c, f]
            """
            pad = N - L
            t = h2
            for axis in range(2*D):  # pad all spatial axes (the first 2*D axes)
                # Compute the shape for the pad tensor to be appended at the end
                pad_shape_post = tf.concat([
                    tf.shape(t)[:axis],
                    [pad],
                    tf.shape(t)[axis+1:]
                ], axis=0)
                pad_tensor_post = tf.zeros(pad_shape_post, dtype=t.dtype)
                t = tf.concat([t, pad_tensor_post], axis=axis)
                # No pad at the beginning (left), only at the end (right)
            return t
        
        if self.dim in [1,2]:
            ## new update ND (tf.pad limitation!)
            # Pad to (N, N, channels)
            # paddings = [
            #     [0, self.N - self.L],
            #     [0, self.N - self.L],
            #     [0, 0],
            #     [0, 0]
            # ]       
            paddings = [[0, self.N - self.L] for _ in range(self.dim*2)]  # spatial dims
            paddings += [[0, 0], [0, 0]]  # in_channels, out_channels
            h2_padded = tf.pad(h2, paddings, mode='CONSTANT', constant_values=0)
        elif self.dim >=3: ## use custom padding if dim 3 or above
            h2_padded = tf_pad_nd_volterra(self.h2, self.N, self.L, self.dim)
        print('h2_padded', h2_padded.shape, h2_padded.dtype)

                
        def get_nd_shifted_h2(h2):
            """Create shifted h2 for vectorized trace convolution"""
            # h2: ([N]*D)*2 + [channels, filters]  (i.e., [N,...,N, N,...,N, c, f])
            # N: spatial size
            # D: number of spatial dimensions
            N = self.N
            D = self.dim

            # Output indices
            n = [tf.range(N) for _ in range(D)]
            k = [tf.range(N) for _ in range(D*2)]

            mesh = tf.meshgrid(*(n + k), indexing='ij')  # mesh of shape [N,...,N]*D + [N,...,N]*2D

            # For each spatial axis, build the shifted indices for both quadratic groups
            idx_list = []
            for d in range(D):
                n_d = mesh[d]
                k1_d = mesh[D + d]
                k2_d = mesh[D + D + d]
                idx_list.append((n_d - k1_d) % N)
            for d in range(D):
                n_d = mesh[d]
                k2_d = mesh[D + D + d]
                idx_list.append((n_d - k2_d) % N)

            idx = tf.stack(idx_list, axis=-1)  # shape: [N,...,N]*3D, 2D
            idx_flat = tf.reshape(idx, [-1, 2*D])
            gathered = tf.gather_nd(h2, idx_flat)
            output_shape = [N] * (3 * D) + [h2.shape[-2], h2.shape[-1]]
            gathered = tf.reshape(gathered, output_shape)
            return gathered    
        
        h2_shifted = get_nd_shifted_h2(h2_padded)
        print('h2shifted', h2_shifted.shape)

        # self.channels = input_shape[-1] # number of channels
        # # Multi-channel kernel: (L, channels)
        # kernel_shape = (self.L, self.channels, self.filters)
        # h = self.add_weight(
        #     # shape=(self.L, self.channels),
        #     shape=kernel_shape,
        #     initializer='glorot_uniform',
        #     trainable=True,
        #     name='UIR_kernel'
        # )
        
        # # Quadratic kernel outer product: (L, channels) x (L, channels) -> (L, L, channels)
        # h2 = tf.einsum('ico,jco->ijco', h, h)  # (L, L, channels)
        # self.h2 = h2
        # # Pad to (N, N, channels)
        # paddings = [
        #     [0, self.N - self.L],
        #     [0, self.N - self.L],
        #     [0, 0],
        #     [0, 0]
        # ]
        # h2_padded = tf.pad(h2, paddings, mode='CONSTANT', constant_values=0)
        # # self.h2_padded = h2_padded
        # ## creating h[n-k1,n-k2]
        # # N = Npoint
        # n = tf.range(self.N)
        # # Create circular indices for all possible shifts
        # # row_indices = tf.math.mod(n[:, tf.newaxis] - tf.range(N), L)  # [N, N]
        # row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)
        # h2_rows_shifted = tf.gather(h2_padded, row_indices, axis=0)         # [N, N, N]
        # h2_columns_shifted = tf.gather(h2_rows_shifted, row_indices,   # [N, N, N]
        #                                 axis=2, batch_dims=1)
        # h2_shifted = h2_columns_shifted
        # # print('h2shifted', self.h2_shifted.shape)

        def make_mra_kernel2(h_shifted):            
            shape = tf.shape(h_shifted) # ij kl mn co 
            _ = tf.reshape(h_shifted, [shape[0]*shape[1]*shape[2]*shape[3], shape[4], shape[5], shape[6]*shape[7]])
            _ = DWT2D(self.wave, clean=False)(_) #mn
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            _ = tf.reshape(_, [shape[0]*shape[1], shape[2], shape[3], shape[4]*shape[5]*shape[6]*shape[7]])
            _ = DWT2D(self.wave, clean=False)(_) #kl
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            _ = tf.reshape(_, [shape[0], shape[1], shape[2]*shape[3], shape[4]*shape[5]*shape[6]*shape[7]])
            _ = tf.transpose(_, perm=[2,0,1,3])
            _ = DWT2D(self.wave, clean=False)(_) #ij
            _ = tf.transpose(_, perm=[1,2,0,3])
            H = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]])
            return H
        H = make_mra_kernel2(h2_shifted)
        print('H',H.shape)
        return H

    def call(self, x):
        # x: (batch, N, channels)
        x2 = tf.einsum('bijc,bklc->bijklc', x, x)  # (batch, N, N, channels)
        print('x2', x2.shape)

        def dwt4(x2):
            shape = tf.shape(x2)
            _ = tf.reshape(x2, [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5]])
            print('_',_.shape)
            _ = DWT2D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]])
            _ = tf.transpose(_, perm=[0,3,4,1,2,5])
            _ = tf.reshape(_, [shape[0]*shape[1]*shape[2], shape[3], shape[4], shape[5]])
            _ = DWT2D(self.wave, clean=False)(_)
            _ = tf.reshape(_, [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]])
            return tf.transpose(_, perm=[0,3,4,1,2,5])
        α2 = dwt4(x2)
        print('α2', α2.shape)
        H = self.make_h2()
        ## Fitlering
        β = tf.einsum('ijklmnco, bklmnc->bijo', H, α2)
        print('β', β.shape)

        # print(x2.shape, self.h2_shifted.shape)
        # y = tf.einsum('ijkco, bjkc->bio', self.h2_shifted, x2)
        # y2 = self.idwt1(β)
        y2 = IDWT2D(self.wave, clean=False)(β)
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
    N = 16
    input_shape = (N, N, 1)  # Replace N with the actual size of x
    inputs = tf.keras.Input(shape=input_shape)

    # Create an instance of the custom layer
    # h2 = np.array([[1, 2], 
    #           [1, 3]]).astype(np.float32)
    # H = TraceConv1Dio(kernel=tf.expand_dims(h2, axis=-1))
    # (tf.expand_dims(tf.expand_dims(x2,axis=0),axis=-1))

    #conv1d_filters=32, conv1d_kernel_size=3, 
    #  conv2d_filters=32, conv2d_kernel_size=3)

    # Apply the custom layer to the inputs
    H = QSIVolterra2Dmra(filters=1, kernel_size=4, wave='haar')
    outputs = H(inputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(1e-4))
    model.summary()