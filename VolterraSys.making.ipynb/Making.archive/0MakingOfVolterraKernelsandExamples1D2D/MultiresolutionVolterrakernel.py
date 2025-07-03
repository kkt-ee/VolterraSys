#%% # include ../dirx 
import tensorflow as tf
print(f"TensorFlow version {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
len(gpus)
mylibpath = [
    '/home/kishoretarafdar/bin'
    #'/home/k/PLAYGROUND10GB/SKULSTRIPpaper__'
    ]
import sys
[sys.path.insert(1,_) for _ in mylibpath]
del mylibpath

from tf_select_a_gpu import select_a_gpu
# select_gpu = gpus[gpu_id]
memory_limit = 16#32 #GB
select_a_gpu(gpus, gpu_id = 2, memory_limit=memory_limit)
# del gpu_id, select_a_gpu, select_gpu


#%%
import tensorflow as tf
from TFDWT3D.DWTIDWT1Dtfv1 import DWT1D, IDWT1D


class TraceConv1Dio(tf.keras.layers.Layer):
    """Trace Convolution: Covolution with strides along the trace of the tensor

       Wiener System / LN cascade with Volterra series
       Input: Sequence x[n]
       Output: Shift invariant quadratic monomial q_2, i.e., m=2

       --@KKT, 7-Feb-2025
    
    """
    
    def __init__(self, kernel_size=4, **kwargs):
        super(TraceConv1Dio, self).__init__(**kwargs)
        self.L = kernel_size
    
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
        ## FULL LENGTH FILTER: kernel_size = input_size
        n = tf.range(self.N)
        # self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)
        
        ## SMALL LENGTH FILTER: kernel_size < input_size
        self.row_indices = tf.math.mod(n[:, tf.newaxis] - tf.range(self.L), self.L)
        super(TraceConv1Dio, self).build(input_shape)
    
    def call(self, x2, h2):
        """
        Args:
            inputs: Tensor of shape (batch_size, N, N, channels)
        Returns:
            Tensor of shape (batch_size, N)
        """
        self.h2 = h2
        print('xh+',x2.shape, self.h2.shape)
        batch_size = tf.shape(x2)[0]
        # print('r+',self.row_indices.shape)
        # Expand indices for batch dimension
        batch_indices = tf.expand_dims(self.row_indices, 0)  # (1, N, N)
        # print('bi+',batch_indices.shape)
        batch_indices = tf.tile(batch_indices, [batch_size, 1, 1])  # (batch, N, N)
        # print('bi+',batch_indices.shape)

        # Gather shifted rows and columns
        rows_shifted = tf.gather(x2, self.row_indices, axis=1)  # (batch, N, N, N, ch)
        columns_shifted = tf.gather(
            rows_shifted, 
            batch_indices,  # Use batch-aware indices
            axis=3, 
            batch_dims=2    # Match batch dimension
        )  # (batch, N, N, N, ch)
        print('rc+',rows_shifted.shape, columns_shifted.shape, self.h2.shape)
 
        result = tf.einsum('bnijc,ijc->bnc', columns_shifted, self.h2)
        return result
    
    def get_config(self):
        base_config = super(TraceConv1Dio, self).get_config()
        return base_config

# import numpy as np
# x = np.array([1, 2, 3,4,5])
# h2 = np.array([[1, 2], 
#               [1, 3]])
# # h2 = np.array([[1, 1, -1], 
# #               [1, 2, -1], 
# #               [1, 3, -1]])

# x2 = np.einsum('i,j->ij', x, x)
# TraceConv1Dio(kernel_size=2)(tf.expand_dims(tf.expand_dims(x2,axis=0),axis=-1), tf.expand_dims(h2, axis=-1))

##%%



class DWTlv1VolterraKernel1Dio(tf.keras.layers.Layer):
    """Multiresolution Volterra Kernel

        Sequence input x[n]

        DWT Lv 1 basis for both 



       --@KKT, 7-Feb-2025
    """
    def __init__(self, kernel_size, Ψ='db4', **kwargs):
        super(DWTlv1VolterraKernel1Dio, self).__init__(**kwargs)
        self.kernel_size = kernel_size  # Tuple (kernel_height, kernel_width)
        # self.out_channels = out_channels
        self.Ψ = Ψ 

    def build(self, input_shape):
        N = input_shape[1]
        in_channels = input_shape[-1]
      
        # Initialize vectors for height and width directions
        self.h = self.add_weight(
            name='kernel_h',
            shape=(self.kernel_size, in_channels),#, self.out_channels),
            initializer='glorot_uniform',
            trainable=True
        )

        """support for orthonormal basis for kernel"""
        self.ϕ_l = self.add_weight(
            name='kernel_ϕ_l',
            shape=(N//2, int(in_channels*2)),
            initializer='glorot_uniform',
            trainable=True
        )

        """support for orthonormal basis for y[n]"""
        self.Φ_v = [
            self.add_weight(
                name=f'kernel_ϕ_{v}',
                shape=(N//2, int(in_channels*2)),
                initializer='glorot_uniform',
                trainable=True)
                for v in range(N//2)
                ]

        self.TraceConv1Dio = TraceConv1Dio(kernel_size=self.kernel_size)
        super(DWTlv1VolterraKernel1Dio, self).build(input_shape)

    def call(self, x):
        ## DWT(x) \times DWT(x)
        self.α = DWT1D(wave=self.Ψ)(x) # batch, N/2, subbands
        
        
        # print('α2+',α2.shape, tf.einsum('bic,bjc->bijc', self.α, self.α).shape)
        ## Loop for v
        # for v in range(N/2):
        #     ϕ_v_inv2 = Φ_v_inv2[v]
        
        
        # Compute outer product to form the 2D kernel        
        # h_exp = tf.expand_dims(self.h, axis=1)  # Shape: (h, 1, in, out)
        # kernel_w_exp = tf.expand_dims(self.kernel_w, axis=0)  # Shape: (1, w, in, out)
        # h2 = h_exp * h_exp  # Broadcasted to (h, w, in, out)
        h2 = tf.einsum('ik,jk->ijk',self.h,self.h)
        print('h+', h2.shape)
        # h2 = tf.transpose(h2, perm=[3,0,1,2]) #(out, h, w, in)
        # print('h+', h2.shape)

        ## DOUBLE IDWT of ϕ_v
        def prepare_Φ_v(ϕ_v):
            # ϕ_v = tf.transpose(ϕ_v, perm=[2,0,1]) # out, N/2, 4.input
            ϕ_v = tf.expand_dims(ϕ_v, axis=0)
            ϕ_v_inv = IDWT1D(wave=self.Ψ)(ϕ_v)         # out, N, 1.input
            ϕ_v = DWT1D(wave=self.Ψ)(ϕ_v_inv)          # out, N/2, 4.input
            ϕ_v_inv2 = IDWT1D(wave=self.Ψ)(ϕ_v)         # out, N, 1.input
            return ϕ_v_inv2
        Φ_v = [prepare_Φ_v(ϕ_v) for ϕ_v in self.Φ_v]
        # print(Φ_v[0].shape)

        

        ## IDWT of ϕ_l
        # ϕ_l = tf.transpose(self.ϕ_l, perm=[2,0,1]) # out, L/2, 4.input
        # print('ϕ_l+', self.ϕ_l.shape, tf.expand_dims(self.ϕ_l,axis=0).shape)
        ϕ_l_inv = IDWT1D(wave=self.Ψ)(tf.expand_dims(self.ϕ_l,axis=0))         # out, L, 1.input
        ## Tensor product of IDWT(kernel_w) with self
        # print('ϕ_l_inv+', ϕ_l_inv.shape)
        Φ2 = tf.einsum('pik,pjk->pijk', ϕ_l_inv, ϕ_l_inv)   # out, L, L, 1.input  
        
        print('T+', Φ2.shape, h2.shape)
        ## Diagonal convolution with h2
        self.h_hash = self.TraceConv1Dio(Φ2, h2) # out, N, 1.input
        print('h_hash+',self.h_hash.shape)
        print('Φ_v[0]+',Φ_v[0].shape)


        # H2 = tf.concat([self.TraceConv1Dio(Φ2, h2) * ϕ_v_inv2 for ϕ_v_inv2 in Φ_v], axis=-1)
        # print('>', H_hash.shape)
        # H2[...,v] * α2 for v in range(H2.shape[-1])


        # α2 = tf.einsum('bic,bjc->bijc', self.α, self.α)
        # print('=',α2.shape)
        # def compute_β(ϕ_v_inv2):
        
        #     # _ = tf.einsum('oni,oni->oni', self.h_hash, ϕ_v_inv2) # out, N, 1.input
        #     _ = self.h_hash * ϕ_v_inv2
        #     H2_l1l2 = tf.expand_dims(tf.reduce_sum(_), axis=-1)#'oni->oi', _) # out, 1.input
        #     print('-',H2_l1l2.shape)#, self.α.shape) 
        #     # β_v = tf.einsum('bic,bjc,ij->bc', self.α, self.α, H2_l1l2) 
        #     # β_v = α2 * H2_l1l2
        #     # print('β_v+',β_v.shape)
        #     # β_v = tf.reduce_sum(β_v, axis=-1, keepdims=True)
        #     # print('β_v+',tf.expand_dims(β_v,axis=1).shape) 
        #     return H2_l1l2
        # # self.β = [compute_β(ϕ_v_inv2) for ϕ_v_inv2 in Φ_v]
        # H2 = tf.concat([tf.expand_dims(compute_β(ϕ_v_inv2), axis=1) for ϕ_v_inv2 in Φ_v], axis=1)
        # print('H2+',H2.shape)
        # self.β = α2 * H2
        # print('β+',self.β.shape) 
        # # β batch, N/2, channels*4


        
        ## this line is incorrect as per formula!!!!
        α2 = tf.reduce_sum(tf.einsum('bic,bjc->bijc', self.α, self.α), axis=[1,2])
        def compute_β(ϕ_v_inv2):
            # _ = tf.einsum('oni,oni->oni', self.h_hash, ϕ_v_inv2) # out, N, 1.input
            _ = self.h_hash * ϕ_v_inv2
            H2_l1l2 = tf.reduce_sum(_)#'oni->oi', _) # out, 1.input
            # print('-',H2_l1l2.shape)#, self.α.shape) 
            # β_v = tf.einsum('bic,bjc,ij->bc', self.α, self.α, H2_l1l2) 
            β_v = α2 * H2_l1l2
            print('β_v+',β_v.shape)
            # β_v = tf.reduce_sum(β_v, axis=-1, keepdims=True)
            # print('β_v+',tf.expand_dims(β_v,axis=1).shape) 
            return β_v
        # self.β = [compute_β(ϕ_v_inv2) for ϕ_v_inv2 in Φ_v]
        self.β = tf.concat([tf.expand_dims(compute_β(ϕ_v_inv2), axis=1) for ϕ_v_inv2 in Φ_v], axis=1)
        print('β+',self.β.shape) 
        # β batch, N/2, channels*4
        
        
        """ multiresolution shift invariant quadratic (m=2) monomial for sequence input
            to an NLSI Wiener system with Volterra series of degree 2
        """
        y_2 = IDWT1D(wave=self.Ψ)(self.β) # batch, N, channels
        
        return y_2


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config

x = tf.random.normal([10, 16, 12])
x = tf.random.normal([1, 16, 1])
# x = tf.random.normal([1, 3, 1])
y = DWTlv1VolterraKernel1Dio(kernel_size=5, Ψ='haar')(x)
y, y.shape
## reduce kernel size than signal size # DONE
## now check with determined inputs











#%%
import tensorflow as tf

class SeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, out_channels, strides=(1, 1), padding='VALID', **kwargs):
        super(SeparableConv2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size  # Tuple (kernel_height, kernel_width)
        self.out_channels = out_channels
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        in_channels = input_shape[-1]
        h, w = self.kernel_size

        # Initialize vectors for height and width directions
        self.kernel_h = self.add_weight(
            name='kernel_h',
            shape=(h, in_channels, self.out_channels),
            initializer='glorot_uniform',
            trainable=True
        )
        self.kernel_w = self.add_weight(
            name='kernel_w',
            shape=(w, in_channels, self.out_channels),
            initializer='glorot_uniform',
            trainable=True
        )
        super(SeparableConv2D, self).build(input_shape)

    def call(self, inputs):
        # Compute outer product to form the 2D kernel
        kernel_h_exp = tf.expand_dims(self.kernel_h, axis=1)  # Shape: (h, 1, in, out)
        kernel_w_exp = tf.expand_dims(self.kernel_w, axis=0)  # Shape: (1, w, in, out)
        kernel = kernel_h_exp * kernel_w_exp  # Broadcasted to (h, w, in, out)

        # Perform standard 2D convolution with the separable kernel
        output = tf.nn.conv2d(
            inputs,
            filters=kernel,
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper() if isinstance(self.padding, str) else self.padding
        )
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'out_channels': self.out_channels,
            'strides': self.strides,
            'padding': self.padding
        })
        return config



#%%
import tensorflow as tf
from TFDWT3D.DWTIDWT1Dtfv1 import DWT1D, IDWT1D


class DWTlv1VolterraKernel1Dio(tf.keras.layers.Layer):
    """Multiresolution Volterra Kernel

        Sequence input x[n]

        DWT Lv 1 basis for both 



       --@KKT, 7-Feb-2025
    """
    def __init__(self, kernel_size, Ψ='db4', **kwargs):
        super(DWTlv1VolterraKernel1Dio, self).__init__(**kwargs)
        self.kernel_size = kernel_size  # Tuple (kernel_height, kernel_width)
        # self.out_channels = out_channels
        self.Ψ = Ψ 

    def build(self, input_shape):
        N = input_shape[1]
        in_channels = input_shape[-1]
      
        # Initialize vectors for height and width directions
        self.h = self.add_weight(
            name='kernel_h',
            shape=(self.kernel_size, in_channels),#, self.out_channels),
            initializer='glorot_uniform',
            trainable=True
        )

        """support for orthonormal basis for kernel"""
        self.ϕ_l = self.add_weight(
            name='kernel_ϕ_l',
            shape=(N//2, int(in_channels*2)),
            initializer='glorot_uniform',
            trainable=True
        )

        """support for orthonormal basis for y[n]"""
        self.Φ_v = [
            self.add_weight(
                name=f'kernel_ϕ_{v}',
                shape=(N//2, int(in_channels*2)),
                initializer='glorot_uniform',
                trainable=True)
                for v in range(N//2)
                ]

        self.TraceConv1Dio = TraceConv1Dio(kernel_size=self.kernel_size)
        super(DWTlv1VolterraKernel1Dio, self).build(input_shape)

    def call(self, x):
        ## DWT(x) \times DWT(x)
        self.α = DWT1D(wave=self.Ψ)(x) # batch, N/2, subbands
        
        
        # print('α2+',α2.shape, tf.einsum('bic,bjc->bijc', self.α, self.α).shape)
        ## Loop for v
        # for v in range(N/2):
        #     ϕ_v_inv2 = Φ_v_inv2[v]
        
        
        # Compute outer product to form the 2D kernel        
        # h_exp = tf.expand_dims(self.h, axis=1)  # Shape: (h, 1, in, out)
        # kernel_w_exp = tf.expand_dims(self.kernel_w, axis=0)  # Shape: (1, w, in, out)
        # h2 = h_exp * h_exp  # Broadcasted to (h, w, in, out)
        h2 = tf.einsum('ik,jk->ijk',self.h,self.h)
        print('h+', h2.shape)
        # h2 = tf.transpose(h2, perm=[3,0,1,2]) #(out, h, w, in)
        # print('h+', h2.shape)

        ## DOUBLE IDWT of ϕ_v
        def prepare_Φ_v(ϕ_v):
            # ϕ_v = tf.transpose(ϕ_v, perm=[2,0,1]) # out, N/2, 4.input
            ϕ_v = tf.expand_dims(ϕ_v, axis=0)
            ϕ_v_inv = IDWT1D(wave=self.Ψ)(ϕ_v)         # out, N, 1.input
            ϕ_v = DWT1D(wave=self.Ψ)(ϕ_v_inv)          # out, N/2, 4.input
            ϕ_v_inv2 = IDWT1D(wave=self.Ψ)(ϕ_v)         # out, N, 1.input
            return ϕ_v_inv2
        Φ_v = [prepare_Φ_v(ϕ_v) for ϕ_v in self.Φ_v]
        # print(Φ_v[0].shape)

        

        ## IDWT of ϕ_l
        # ϕ_l = tf.transpose(self.ϕ_l, perm=[2,0,1]) # out, L/2, 4.input
        # print('ϕ_l+', self.ϕ_l.shape, tf.expand_dims(self.ϕ_l,axis=0).shape)
        ϕ_l_inv = IDWT1D(wave=self.Ψ)(tf.expand_dims(self.ϕ_l,axis=0))         # out, L, 1.input
        ## Tensor product of IDWT(kernel_w) with self
        # print('ϕ_l_inv+', ϕ_l_inv.shape)
        Φ2 = tf.einsum('pik,pjk->pijk', ϕ_l_inv, ϕ_l_inv)   # out, L, L, 1.input  
        
        print('T+', Φ2.shape, h2.shape)
        ## Diagonal convolution with h2
        self.h_hash = self.TraceConv1Dio(Φ2, h2) # out, N, 1.input
        print('h_hash+',self.h_hash.shape)
        print('Φ_v[0]+',Φ_v[0].shape)


        # H2 = tf.concat([self.TraceConv1Dio(Φ2, h2) * ϕ_v_inv2 for ϕ_v_inv2 in Φ_v], axis=-1)
        # print('>', H_hash.shape)
        # H2[...,v] * α2 for v in range(H2.shape[-1])


        # α2 = tf.einsum('bic,bjc->bijc', self.α, self.α)
        # print('=',α2.shape)
        # def compute_β(ϕ_v_inv2):
        
        #     # _ = tf.einsum('oni,oni->oni', self.h_hash, ϕ_v_inv2) # out, N, 1.input
        #     _ = self.h_hash * ϕ_v_inv2
        #     H2_l1l2 = tf.expand_dims(tf.reduce_sum(_), axis=-1)#'oni->oi', _) # out, 1.input
        #     print('-',H2_l1l2.shape)#, self.α.shape) 
        #     # β_v = tf.einsum('bic,bjc,ij->bc', self.α, self.α, H2_l1l2) 
        #     # β_v = α2 * H2_l1l2
        #     # print('β_v+',β_v.shape)
        #     # β_v = tf.reduce_sum(β_v, axis=-1, keepdims=True)
        #     # print('β_v+',tf.expand_dims(β_v,axis=1).shape) 
        #     return H2_l1l2
        # # self.β = [compute_β(ϕ_v_inv2) for ϕ_v_inv2 in Φ_v]
        # H2 = tf.concat([tf.expand_dims(compute_β(ϕ_v_inv2), axis=1) for ϕ_v_inv2 in Φ_v], axis=1)
        # print('H2+',H2.shape)
        # self.β = α2 * H2
        # print('β+',self.β.shape) 
        # # β batch, N/2, channels*4


        
        ## this line is incorrect as per formula!!!!
        α2 = tf.reduce_sum(tf.einsum('bic,bjc->bijc', self.α, self.α), axis=[1,2])
        def compute_β(ϕ_v_inv2):
            # _ = tf.einsum('oni,oni->oni', self.h_hash, ϕ_v_inv2) # out, N, 1.input
            _ = self.h_hash * ϕ_v_inv2
            H2_l1l2 = tf.reduce_sum(_)#'oni->oi', _) # out, 1.input
            # print('-',H2_l1l2.shape)#, self.α.shape) 
            # β_v = tf.einsum('bic,bjc,ij->bc', self.α, self.α, H2_l1l2) 
            β_v = α2 * H2_l1l2
            print('β_v+',β_v.shape)
            # β_v = tf.reduce_sum(β_v, axis=-1, keepdims=True)
            # print('β_v+',tf.expand_dims(β_v,axis=1).shape) 
            return β_v
        # self.β = [compute_β(ϕ_v_inv2) for ϕ_v_inv2 in Φ_v]
        self.β = tf.concat([tf.expand_dims(compute_β(ϕ_v_inv2), axis=1) for ϕ_v_inv2 in Φ_v], axis=1)
        print('β+',self.β.shape) 
        # β batch, N/2, channels*4
        
        
        """ multiresolution shift invariant quadratic (m=2) monomial for sequence input
            to an NLSI Wiener system with Volterra series of degree 2
        """
        y_2 = IDWT1D(wave=self.Ψ)(self.β) # batch, N, channels
        
        return y_2


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
        })
        return config

x = tf.random.normal([10, 16, 12])
x = tf.random.normal([1, 16, 1])
# x = tf.random.normal([1, 3, 1])
y = DWTlv1VolterraKernel1Dio(kernel_size=5, Ψ='haar')(x)
y, y.shape
## reduce kernel size than signal size # DONE
## now check with determined inputs




