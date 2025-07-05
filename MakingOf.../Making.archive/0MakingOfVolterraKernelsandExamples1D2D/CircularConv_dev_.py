#%% 3D FFT convolution with wavelet regularizer (CHECK with values!!)
import tensorflow as tf

class wL1L2Regularizer(tf.keras.Regularizer):
    
  def __init__(self, l1=0., l2=0., Ψ='haar'):
    self.l1 = l1
    self.l2 = l2
    self.Ψ = Ψ

  def __call__(self, x):
    # wx = DWT2D(wave=self.Ψ)(tf.einsum('ijkl->kijl', x))
    # wx = dwtlv4(tf.einsum('ijkl->kijl', x), Ψ='haar')
    wx = dwt3dlv3(tf.einsum('mijkl->kmijl', x), Ψ='haar')
    c_L1wx = self.l1 * ops.sum([ops.sum(ops.abs(_)) for _ in wx])
    c_L2wx = self.l2 * ops.sum([ops.sum(ops.square(_)) for _ in wx])
    # c_L1wx = self.l1 * ops.sum(ops.abs(wx))
    # c_L2wx = self.l2 * ops.sum(ops.square(wx))
    return c_L1wx + c_L2wx 
    # self.l1 * ops.sum(ops.square(wx))

  def get_config(self):
    return {
        'l1': float(self.l1),
        'l2': float(self.l2)
    }


def dwt3dlv3(x, Ψ='haar'):
    # c = 1
    w1 = DWT3D(wave=Ψ)(x)
    l1 = w1[:,:,:,:,:1]
    h1 = w1[:,:,:,:,1:]
    # print(l1.shape)
    #
    w2 = DWT3D(wave=Ψ)(l1)
    l2 = w2[:,:,:,:,:1]
    h2 = w2[:,:,:,:,1:]
    #
    w3 = DWT3D(wave=Ψ)(l2)
    l3 = w3[:,:,:,:,:1]
    h3 = w3[:,:,:,:,1:]
    #
    # w4 = DWT2D(wave=Ψ)(l3)
    # l4 = w4[:,:,:,:1]
    # h4 = w4[:,:,:,1:]
    #
    return h1, h2, h3 

def dwt3dlv4(x, Ψ='haar'):
    # c = 1
    w1 = DWT3D(wave=Ψ)(x)
    l1 = w1[:,:,:,:,:1]
    h1 = w1[:,:,:,:,1:]
    # print(l1.shape)
    #
    w2 = DWT3D(wave=Ψ)(l1)
    l2 = w2[:,:,:,:,:1]
    h2 = w2[:,:,:,:,1:]
    #
    w3 = DWT3D(wave=Ψ)(l2)
    l3 = w3[:,:,:,:,:1]
    h3 = w3[:,:,:,:,1:]
    #
    w4 = DWT3D(wave=Ψ)(l3)
    l4 = w4[:,:,:,:,:1]
    h4 = w4[:,:,:,:,1:]
    #
    return h1, h2, h3, h4, l4 
    


## check with some numbers and multiplication property
class FFTConv3D(tf.keras.layers.Layer):
    """3D circular convolution layer with tf.signal.fft3d"""
    def __init__(self, filters, filter_size=9, l1=0.0, l2=0.0):
        super(FFTConv3D, self).__init__()
        self.filters = filters
        self.filter_size = filter_size
        # self.kernel_regularizer=tf.keras.regularizers.L1L2(l1, l2)
        # self.kernel_regularizer=wL1Regularizer(l1=1e-5, Ψ='haar')
        # self.kernel_regularizer=wL2Regularizer(l2=1e-4, Ψ='haar')
        self.kernel_regularizer=wL1L2Regularizer(l1=1e-5, l2=1e-4, Ψ='haar')

    def build(self, input_shape):
        # Initialize the kernel with the specified shape
        kernel_shape = (self.filter_size, self.filter_size, self.filter_size, input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True,
            regularizer=self.kernel_regularizer,
            name='kernel'
        )
        # self.kernel_copies = tf.stack([self.kernel] * input_shape[-1], axis=-1)
        # self.kernel = tf.cast(self.kernel, tf.complex64)
    
    def call(self, x):
        Npoint = x.shape[1]
        # print(x.shape, self.kernel.shape, x.dtype, self.kernel.dtype)
        x = tf.transpose(x,perm=[0,4,1,2,3])
        X = tf.signal.fft3d(tf.cast(x, tf.complex64))
        # X = tf.transpose(X,perm=[0,2,3,4,1])
        

        # x_fft = tf.fftnd(tf.cast(x, tf.complex64), fft_length=[Npoint, Npoint, Npoint], axes=[1, 2, 3])

        
        paddings = [
            [0, Npoint - self.filter_size],  # Depth dimension
            [0, Npoint - self.filter_size],  # Height dimension
            [0, Npoint - self.filter_size],  # Width dimension
            [0, 0],
            [0, 0]  # Batch dimension (no padding)
        ]
        h_padded = tf.pad(self.kernel, paddings, mode='CONSTANT', constant_values=0)
        h_padded = tf.transpose(h_padded,perm=[4,3,0,1,2])
        #print(self.kernel.shape,kernel_padded.shape, '+')
        H = tf.signal.fft3d(tf.cast(h_padded, tf.complex64))
        # H = tf.transpose(H,perm=[2,3,4,1,0])
        #print(x_fft.shape, kernel_fft.shape, x_fft.dtype, kernel_fft.dtype)
        
        # Element-wise multiplication in the frequency domain
        # Y = tf.einsum('bmijk,mijkl->bmijl', X, H)
        Y = tf.einsum('blijk,mlijk->bmijk', X, H)
        # print('fft mul', product.shape)
        # Compute the inverse FFT to get back to the spatial domain
        # print('+',product.shape, product.dtype)
        # Y = tf.transpose(Y,perm=[0,4,1,2,3])
        y = tf.signal.ifft3d(Y)
        y = tf.transpose(y,perm=[0,2,3,4,1])
        # print('++',product_ifft.shape, product_ifft.dtype)
        y_real = tf.cast(tf.math.real(y), dtype=tf.float32)
        print('ifft shape',y_real.shape, y_real.dtype)
        #return tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        return y_real#tf.cast(product_ifft_real, tf.float32)



    def get_config(self):
        config = super(FFTConv3D, self).get_config()
        return config
    
# Example usage
## ok with filters=1
# inputs = tf.random.normal((1, 2, 2, 2, 1))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)
# inputs = tf.random.normal((1, 3, 3, 3, 1))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)
# inputs = tf.random.normal((1, 28, 28, 28, 1))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)

inputs = tf.random.normal((1, 2, 2, 2, 3))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)
inputs = tf.random.normal((1, 28, 28, 28, 4))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)
inputs = tf.random.normal((4, 28, 28, 28, 4))  # Example input with shape (batch_size=1, height=28, width=28, channels=1)


# Create the custom Conv2D layer with 16 filters and a 3x3 kernel size
custom_conv_layer = FFTConv3D(filters=1, filter_size=2)
# custom_conv_layer = FFTConv3D(filters=16, filter_size=3)
output = custom_conv_layer(inputs)
    
print('+out', output.shape)  # The shape depends on input size, kernel size, strides, and padding
output



#%% N-dim. circular convolution with tf.signal.fftnd (tested) (make layer!)
# Here N=3
x = inputs
h = custom_conv_layer.kernel.numpy()
x.shape, h.shape

x = tf.cast(x, dtype=tf.complex64)
h = tf.cast(h, dtype=tf.complex64)
Npoint = x.shape[1]
paddings = [
            [0, Npoint - h.shape[0]],  # Depth dimension
            [0, Npoint - h.shape[0]],  # Height dimension
            [0, Npoint - h.shape[0]],  # Width dimension
            [0, 0],
            [0, 0]  # Batch dimension (no padding)
        ]
h = tf.pad(h, paddings, mode='CONSTANT', constant_values=0)



x = tf.transpose(x,perm=[0,4,1,2,3])
print('x', x.shape)
X = tf.signal.fftnd(x, fft_length=[Npoint,Npoint,Npoint], axes=[-3,-2,-1])  # 3D FFT of x
print('X', X.shape)
h = tf.transpose(h,perm=[4,3,0,1,2])
print('h', h.shape)
H = tf.signal.fftnd(h, fft_length=[Npoint,Npoint,Npoint], axes=[-3,-2,-1])  # 3D FFT of h
print('H', H.shape)

# Element-wise multiplication of the FFTs
Y = X * H
Y = tf.einsum('blijk,mlijk->bmijk', X, H)
print(X.shape, H.shape, Y.shape)
# Compute the inverse 3D Fourier Transform
y = tf.signal.ifftnd(Y, fft_length=[Npoint,Npoint,Npoint], axes=[-3,-2,-1])
# print('y', y.shape)
y = tf.transpose(y,perm=[0,2,3,4,1])
# print('y', y.shape)

# The result should be real (ignoring small imaginary parts due to numerical errors)
ifft_result_ = np.real(y)

# Print the results
# print("Convolution result (scipy):\n", conv_result)
# print(Y.shape)
print("Inverse FFT of product of FFTs:\n", ifft_result_.shape, ifft_result_)

# Check if the results are approximately equal
print("Are the results equal?", np.allclose(output, ifft_result_, atol=1e-10))
# print("Are the results equal?", np.allclose(tf.squeeze(output), ifft_result_, atol=1e-10))

