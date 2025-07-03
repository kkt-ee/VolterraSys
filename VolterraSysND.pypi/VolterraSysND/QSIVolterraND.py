import tensorflow as tf
import string

@tf.keras.utils.register_keras_serializable()
class QSIVolterraND(tf.keras.layers.Layer):
    """ QSI Volterra ND [1D, 2D, 3D] kernels in natural domain

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
    """
    def __init__(self, filters=1, kernel_size=4, **kwargs):
        super().__init__(**kwargs)
        self.L = kernel_size
        self.filters = filters

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
        if self.dim==1:
            h2 = tf.einsum('ico,jco->ijco', self.h, self.h)  # (L, L, channels)       
        elif self.dim==2:
            h2 = tf.einsum('ijco,klco->ijklco', self.h, self.h)  # (L, L, channels)
        elif self.dim==3:
            h2 = tf.einsum('ijkco,pqrco->ijkpqrco', self.h, self.h)  # (L, L, channels)
        else:
            raise NotImplementedError("Hardcoded einsum strings supported for dim=1,2,3")
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
        return h2_shifted
        

    def get_einsum_strings(self, dim):
        """
        Returns einsum strings for quadratic Volterra layer, hardcoded for dim=1,2,3,
        matching the exact manual pattern.
        """
        if dim == 1:
            x2_eq = 'bic,bjc->bijc'
            y2_eq = 'ijkco,bjkc->bio'
        elif dim == 2:
            x2_eq = 'bijc,bklc->bijklc'
            y2_eq = 'ijklmnco, bklmnc->bijo'
        elif dim == 3:
            x2_eq = 'bijkc,bpqrc->bijkpqrc'
            y2_eq = 'ijkpqrstuco,bpqrstuc->bijko'
        else:
            raise NotImplementedError("Hardcoded einsum strings supported for dim=1,2,3")
        return x2_eq, y2_eq       

    def call(self, x):
        x_shape = x.shape  # [batch, N, ..., N, channels]
        # x2_eq, y2_eq = self.get_einsum_strings(self.dim)
        # print('str1 ', x2_eq, '\nstr2 ', y2_eq)
        # x2: (batch, N^dim, N^dim, channels)
        x2_eq, y2_eq = self.get_einsum_strings(self.dim)
        x2 = tf.einsum(x2_eq, x, x)
        h2_shifted = self.make_h2()
        # y2: (batch, N^dim, output_channels)
        y2 = tf.einsum(y2_eq, h2_shifted, x2)
        return y2       

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
    H = QSIVolterraND(filters=filters)
    outputs = H(inputs)
    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()
    ## 1D Random data
    x = tf.random.normal((1, N, 1))
    y = tf.random.normal((1, N, 1))
    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(x, y, epochs=5, verbose=1)

    ## OR 
    model.predict(x)


"""
2D I/O
h2_padded (4, 4, 4, 4, 1, 1) <dtype: 'float32'>
h2shifted (4, 4, 4, 4, 4, 4, 1, 1)
> (4, 4, 4, 4, 4, 4, 1, 1) (None, 4, 4, 4, 4, 1)

"""