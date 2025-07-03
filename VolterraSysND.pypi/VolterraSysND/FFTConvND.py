#%% Checked equivalence of natural and fft domain
## Equivalent to cyclic convolution
import tensorflow as tf

class FFTConvND(tf.keras.layers.Layer):
    """ ND convolutions using convolution theorem 
        Separable FFTND

        Equivalent to circular convolution

        tf.pad limitation: Only ranks up to 6 supported: [2,2,2,2,2,2,4] [Op:Pad]
        Therefore this layer supports upto 5D convs!!
        
        patch update tf_pad_nd method for 6D 
        Current limitation: einsum table hardcoded till 6D conv

        Note: there will be error if L>N


        VolterraSys: Multidimensional linear and nonlinear Volterra kernels in natural and biortogonal bases.
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
        
    --kkt@07-Jun-2025"""
    def __init__(self, filters, kernel_size=9):#, l1=0.0, l2=0.0):
        super(FFTConvND, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        # self.dim = dim
        # self.axes = axes

   
    def build(self, input_shape):
        self.N = input_shape[1] ## !!hyper square inputs only
        self.axes = list(range(1, len(input_shape) - 1))
        self.dim = self.axes[-1]
        in_channels = input_shape[-1] 
        
        # Initialize the kernel with the specified shape
        ##OPTION 1: NO PADDING and L=N by default
        # *input_shape[]
        # kernel_size = input_shape[1:self.dim+1] 
        # kernel_shape = kernel_size + (input_shape[-1], self.filters)      

        ##OPTION 2: [update applied] for smaller size kernels
        kernel_shape = [self.kernel_size] * self.dim + [in_channels, self.filters]
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        
    def call(self, x):
        print(x.shape, self.kernel.shape, x.dtype, self.kernel.dtype)
        ## FFT of input and kernel 
        x_fft = self.__fftnd(tf.cast(x, tf.complex64), axes=self.axes)
        
        ##OPTION 1: NO PADDING and L=N by default
        # kernel_fft = self.__fftnd(tf.cast(self.kernel, tf.complex64), axes=self.axes)
        
        ##OPTION 2: [update applied] for smaller size kernels
        kernel = self._pad_kernel_to_match_input_shape()
        
        ## fixed here n,n,i,o ----- dynamic perm based in input dim---
        perm1, perm2 = self._get_kernel_perms()
        kernel = tf.transpose(kernel,perm=perm1) ## o,n,n,i
        kernel_fft = self.__fftnd(tf.cast(kernel, tf.complex64), axes=self.axes)
        kernel_fft = tf.transpose(kernel_fft,perm=perm2) ## n,n,i,o        
        ##-----------------------------------------------
              
        
        print('<>' ,x_fft.shape, kernel_fft.shape, x_fft.dtype, kernel_fft.dtype)
        print(tf.squeeze(x), tf.squeeze(x_fft), '\nkernel', tf.squeeze(self.kernel), tf.squeeze(kernel_fft))
        
        # Element-wise multiplication in the frequency domain
        print(f'{self.dim} dimensional conv')
        KXproduct = tf.einsum(self._einsum_nd_ops(self.dim), x_fft, kernel_fft)
        print('fft mul KX ', KXproduct.shape, tf.squeeze(KXproduct))
    
        # Compute the inverse FFT to get back to the spatial domain
        print('+axes', KXproduct.shape, KXproduct.dtype, self.axes)
        KXproduct_ifft = self.__ifftnd(KXproduct, axes=self.axes)
        print('+ifft',KXproduct_ifft.shape, KXproduct_ifft.dtype, tf.squeeze(KXproduct_ifft))
        KXproduct_ifft_real = tf.cast(tf.math.real(KXproduct_ifft ), dtype=tf.float32)
        print('+real',KXproduct_ifft_real.shape, KXproduct_ifft_real.dtype, tf.squeeze(KXproduct_ifft_real))
        #return tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        return KXproduct_ifft_real #tf.cast(product_ifft_real, tf.float32)



    def _einsum_nd_ops(self, dim):
        """Support upto 5D dimensional tensor ops"""
        einsum_nd = {
            1:'bik,ikl->bil',                       ## conv1D
            2:'bijk,ijkl->bijl',                    ## conv2D
            3:'bijpk,ijpkl->bijpl',                 ## conv3D
            4:'bijpqk,ijpqkl->bijpql',              ## conv4D
            5:'bijpqrk,ijpqrkl->bijpqrl',           ## conv5D
            
            6:'bijpqrsk,ijpqrskl->bijpqrsl',        ## conv6D solution: manually define tf_pad_nd
            # 7:'bijpqrstk,ijpqrstkl->bijpqrstl',        ## conv7D!! solution: manually define tf.pad
            # 8:'bijpqrstuk,ijpqrstukl->bijpqrstul',     ## conv8D!! solution: manually define tf.pad
            # 9:'bijpqrstuvk,ijpqrstuvkl->bijpqrstuvl',  ## conv9D!! solution: manually define tf.pad
            }
        return einsum_nd[dim]


    def _pad_kernel_to_match_input_shape(self):
        """
        Generate paddings for kernel tensor to match spatial size N from L,
        using input_shape to determine number of spatial dimensions.

        Args:
            input_shape: tf.TensorShape or tuple/list of shape (batch, D1, ..., Dn, in_channels)
            N: target spatial size (assumed same for all dims)
            L: original kernel size (assumed same for all spatial dims)

        Returns padded kernel where,
            paddings: list of [0, N - L] for each spatial dim + [[0, 0], [0, 0]] for in/out channels
        
        
        TF limiation: Only ranks up to 6 supported: [2,2,2,2,2,2,4] [Op:Pad]
        tf_pad_nd overcomes this
        Einsum ops hardcoded upto 6D (spatial) ops
        """
        # @tf.function
        def tf_pad_nd(kernel):
            """
            Equivalent to tf.pad(kernel, paddings) where:
                paddings = [[0, N - kernel_size]] * D + [[0, 0], [0, 0]]
            Works beyond tf.pad's rank-7 limitation.

            Args:
                kernel: tf.Tensor of shape [L1, L2, ..., LD, in_channels, out_channels]
                N: int, target spatial size
                kernel_size: int, original spatial kernel size (assumed same for all dims)
                D: int, number of spatial dimensions

            Returns:
                padded kernel of shape [N]*D + [in_channels, out_channels]
            """
            N = self.N
            kernel_size = self.kernel_size
            D = self.dim
            pad = N - kernel_size
            t = kernel
            rank = tf.rank(kernel)

            for axis in range(D):  # only spatial dims
                pad_shape_post = tf.concat([
                    tf.shape(t)[:axis],
                    [pad],
                    tf.shape(t)[axis + 1:]
                ], axis=0)

                pad_tensor_post = tf.zeros(pad_shape_post, dtype=t.dtype)
                t = tf.concat([t, pad_tensor_post], axis=axis)

            return t

        # num_spatial_dims = len(input_shape) - 2  # Exclude batch and channels
        paddings = [[0, self.N - self.kernel_size] for _ in range(self.dim)]  # spatial dims
        paddings += [[0, 0], [0, 0]]  # in_channels, out_channels
        print('dim', self.dim)
        print(paddings)
        if self.dim in [1,2,3,4,5]:
            print('kernel shape', self.kernel.shape)
            kernel_padded = tf.pad(self.kernel, paddings, mode='CONSTANT', constant_values=0)
            print(f'dim {self.dim} kernel_padded', kernel_padded.shape)
            return kernel_padded
        elif self.dim ==6:
            kernel_padded = tf_pad_nd(self.kernel) # kernel and some args..... may be)
            print(f'dim {self.dim} kernel_padded', kernel_padded.shape)
            return kernel_padded
        else:
            raise NotImplementedError(f"Input spatial dim is {self.dim}!! Spatial dim upto 6 suppored (update hardcoded einsum strings table)") 



    def _get_kernel_perms(self):
        """
        Given input shape: (batch, D1, D2, ..., Dn, in_channels)
        Returns:
            - first_perm: for kernel: (D1, D2, ..., Dn, in_channels, out_channels) → (out_channels, D1, ..., Dn, in_channels)
            - second_perm: to bring it back → (D1, ..., Dn, in_channels, out_channels)
        """
        # num_spatial_dims = len(input_shape) - 2  # Exclude batch and channels



        # perm to move output channels (last) to front
        first_perm = [self.dim + 1] + list(range(self.dim)) + [self.dim]
        
        # perm to move output channels back to end
        second_perm = list(range(1, self.dim + 1)) + [self.dim + 1, 0]

        return first_perm, second_perm

    # input_shape = (None, 8, 8, 8, 8, 1)  # 3D input: batch, D1, D2, D3, channels
    # perm1, perm2 = get_kernel_perms(input_shape)

    def __fftnd(self, x, axes):
        """Compute N-dimensional FFT by applying 1D FFT along specified axes."""
        for axis in axes:
            # Move axis to the last position
            perm = list(range(x.shape.rank))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = tf.transpose(x, perm)
            x = tf.signal.fft(x)
            x = tf.transpose(x, perm)
        return x

    def __ifftnd(self, x, axes):
        """Compute N-dimensional inverse FFT by applying 1D IFFT along specified axes."""
        for axis in axes:
            perm = list(range(x.shape.rank))
            perm[axis], perm[-1] = perm[-1], perm[axis]
            x = tf.transpose(x, perm)
            x = tf.signal.ifft(x)
            x = tf.transpose(x, perm)
        return x


    def get_kernel(self):
        """Returns the kernel weights as a numpy array"""
        return self.kernel.numpy()

    # def compute_output_shape(self, input_shape):
    #     # input_shape is a tf.TensorShape or tuple: (batch, D1, ..., Dn, channels_in)
    #     batch_size = input_shape[0]
    #     spatial_dims = input_shape[1:-1]  # D1,...,Dn
    #     channels_out = self.filters  # assuming self.filters = number of output channels
        
    #     # Output shape preserves spatial dims, changes channels to filters
    #     output_shape = tf.TensorShape([batch_size, *spatial_dims, channels_out])
    #     return output_shape

   
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'filters': self.filters            
        })
        return config


if __name__ =='__main__':
    # Functional model
    N, channels, filters = 4, 1, 1
    # input_shape = (N, channels)  # Replace N with the actual size of x            #1D
    # input_shape = (N, N, channels)  # Replace N with the actual size of x       #2D
    # input_shape = (N, N, N, channels)  # Replace N with the actual size of x    #3D
    # input_shape = (N, N, N, N, channels)  # Replace N with the actual size of x    #4D
    # input_shape = (N, N, N, N, N, channels)  # Replace N with the actual size of x    #5D
    input_shape = (N, N, N, N, N, N, channels)  # Replace N with the actual size of x    #6D

    inputs = tf.keras.Input(shape=input_shape)

    # Create an instance of the custom layer
    # h2 = np.array([[1, 2], 
    #           [1, 3]]).astype(np.float32)
    # H = TraceConv1Dio(kernel=tf.expand_dims(h2, axis=-1))
    # (tf.expand_dims(tf.expand_dims(x2,axis=0),axis=-1))

    #conv1d_filters=32, conv1d_kernel_size=3, 
    #  conv2d_filters=32, conv2d_kernel_size=3)

    # Apply the custom layer to the inputs
    H = FFTConvND(filters=1, kernel_size=1)
    outputs = H(inputs)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', jit_compile=False)
    model.summary()

    # Random data
    ## 1D
    # inputs_data = tf.random.normal((1, N, 1))
    # targets = tf.random.normal((1, N, 1))
    ## 2D
    # inputs_data = tf.random.normal((1, N, N, 1))
    # targets = tf.random.normal((1, N, N, 1))
    ## 3D
    # inputs_data = tf.random.normal((1, N, N, N, 1))
    # targets = tf.random.normal((1, N, N, N, 1))
    ## 4D
    # inputs_data = tf.random.normal((1, N, N, N, N, 1))
    # targets = tf.random.normal((1, N, N, N, N, 1))
    ## 5D
    # inputs_data = tf.random.normal((1, N, N, N, N, N, 1))
    # targets = tf.random.normal((1, N, N, N, N, N, 1))
    ## 6D
    x = tf.random.normal((1, N, N, N, N, N, N, 1))
    y = tf.random.normal((1, N, N, N, N, N, N, 1))


    # Training loop for 5 epochs
    epochs=5
    # for epoch in range(5):
    history = model.fit(x, y, epochs=5, verbose=1)

    ## Example
    model.predict(x)
