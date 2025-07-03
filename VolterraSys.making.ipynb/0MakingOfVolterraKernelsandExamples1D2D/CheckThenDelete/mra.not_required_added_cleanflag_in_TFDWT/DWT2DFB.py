#%% 
### new with einsum operations
"""Fixed the DWT and IDWT matrix.
The matrix is of same dimension as of the signal. No padding is applied.
Circular convolution takes care of the boundary effects. @KKT 14/03/2024 """

# include ../dirx 
# mylibpath = '/home/k/PLAYGROUND10GB/myDTSPlib'
# mylibpath = '/home/k/PLAYGROUND10GB/kerasDWTLayer1D2D_orthogonal_wavelets.conv/FastDWTConvLayers/pypipackageTFDWT3d'
# import sys
# sys.path.insert(0,mylibpath)
# del mylibpath

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import scipy
# import pywt
import numpy as np
# print(f"\
#     \n tf {tf.__version__}, \
#     \n matplotlib {matplotlib.__version__} \
#     \n scipy {scipy.__version__}\
#     \n pywavelets {pywt.__version__} (for comparing tf results)")


# def get_A_matrix_dwt_analysisFB_unit(h0,h1,N:int):
#     """returns Analysis Matrix A"""
#     L = len(h0)
#     __H_branch_row = lambda h0,N: np.concatenate((h0, np.zeros(int(N - len(h0)))))
#     __H_start_row = lambda _row: np.roll(_row, shift=-L+2, axis=None)
#     __H_start_row(__H_branch_row(h0,N))
#     __H_branch= lambda _row: [np.roll(_row, shift=k*2, axis=None) for k in range(_row.shape[0])]#-len(h0)+1)]
#     H0 = __H_branch(__H_start_row(__H_branch_row(h0,N)))
#     H1 = __H_branch(__H_start_row(__H_branch_row(h1,N)))
#     return np.concatenate((H0[:int(len(H0)/2)], H1[:int(len(H1)/2)]))

# A = get_A_matrix_dwt_analysisFB_unit(h0,h1,N)
# A.shape, A


## OK
# import pywt
from TFDWT.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit
@keras.saving.register_keras_serializable()
class DWT2D(tf.keras.layers.Layer):
    """2D DWT layer @kkt"""
    def __init__(self, wave='haar', **kwargs):
        super(DWT2D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        self.h0, self.h1 = w.analysis()
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        # self.padding_mode = 'symmetric'
        # self.N
        # self.A
        
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # self.pw = (self.A.shape[0]-self.N)//2
        super(DWT2D, self).build(input_shape) 
       
    
    def call(self, inputs):
        # ch_ = [self.__pad(inputs[..., i:i+1]) for i in range(self.num_channels)]
        # ch_ = tf.concat(ch_, axis=-1)
        # ch_ = [self.__dwt(ch_[..., i:i+1]) for i in range(self.num_channels)]
        ch_ = [self.__dwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.concat(ch_, axis=-1)
        #
        # out = self.__extract_4subbands(out)
        return out

    @tf.function
    def __dwt(self, x):
        # print('>>',x.shape)
        # x = tf.squeeze(x, axis=-1)
        # _ = tf.transpose(x, perm=[0,2,1])
        # _rowconv_out = tf.cast(self.A, tf.float32)@tf.cast(_, tf.float32)
        # _colconv_out = tf.cast(self.A, tf.float32)@tf.transpose(_rowconv_out, perm=[0,2,1])
        # _LLLHHLHH = _colconv_out
        ## end dwt



        ## einsum code
        _x2d = tf.squeeze(x, axis=-1)
        _x2d = tf.transpose(_x2d,[0,2,1])
        A = tf.cast(self.A, tf.float32)
        _rowconv_out = tf.einsum('ij,bjk->bik', A, _x2d)
        _rowconv_outT = tf.transpose(_rowconv_out,[0,2,1])
        _LLLHHLHH = tf.einsum('ij,bjk->bik', A, _rowconv_outT)
        _colconv_out = _LLLHHLHH
        # _LLLHHLHH.shape
        # plt.imshow(_LLLHHLHH)
        ## einsum code end








        # OK for batch   
        # perm=[1,2,0]#ok but rotate 90
        # __ = tf.cast(self.A, tf.float32)@tf.transpose(_rowconv_out, perm=perm)
       
        # perm=[2,1,0]
        # __ = tf.transpose(__, perm=perm)
        # print('*T',__.shape)
        
        # plt.imshow(__[1,:,:]), plt.show()
        # plt.imshow(__[0,:,:]), plt.show()
        # return __
        return tf.expand_dims(_colconv_out, axis=-1)

    # def __pad(self, x):
    #     x = tf.squeeze(x, axis=-1)
    #     paddings = tf.constant([[0,0], [self.pw, self.pw,], [self.pw, self.pw]])

    #     x_padded = tf.pad(x, paddings, self.padding_mode)
    #     # print('+',x_padded.shape) 
    #     # return x_padded
    #     return tf.expand_dims(x_padded, axis=-1)

    def __extract_4subbands(self,LLLHHLHH):
        """returns 4 image subbands LL, LH, HL, HH from DWT Analysis bank o/p --@k"""
        # global N
        _mid = int(LLLHHLHH.shape[1]/2)
        LL = LLLHHLHH[:,:_mid,:_mid,:]
        LH = LLLHHLHH[:,_mid:,:_mid,:]
        HL = LLLHHLHH[:,:_mid,_mid:,:]
        HH = LLLHHLHH[:,_mid:,_mid:,:]
        # _i = int((LL.shape[1] - self.N/2)/2)
        # out = tf.concat([LL[:,_i:-_i,_i:-_i,:], LH[:,_i:-_i,_i:-_i,:], HL[:,_i:-_i,_i:-_i,:], HH[:,_i:-_i,_i:-_i,:]], axis=-1)
        out = tf.concat([LL, LH, HL, HH], axis=-1)
        return out

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width = height // 2, width // 2
    #     return (batch_size, new_height, new_width, 4 * channels)
    
    def get_config(self):
        config = super(DWT2D, self).get_config()
        return config



# ## WORKING IDWT similar to wavetf
# import pywt
# from GETDWTFiltersOrtho import GETDWTFiltersOrtho
@keras.saving.register_keras_serializable()
class IDWT2D(tf.keras.layers.Layer):
    """2D IDWT layer @kkt"""
    def __init__(self, wave='haar', **kwargs):
        super(IDWT2D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            # self.g0, self.g1 = w.synthesis()
            # self.h0, self.h1 = self.g0, self.g1
            # del self.g0, self.g1
            self.h0, self.h1 = w.synthesis()
            # self.biortho_flag = 1
            # print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        # self.padding_mode = 'symmetric'

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = int(input_shape[1])
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # self.pw = int(self.L/2)
        super(IDWT2D, self).build(input_shape) 

    def call(self, inputs):
        # self.N = int(inputs.shape[1]*2)
        # self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # input_shape = tf.shape(inputs)

        #all input channels
        # ch_ = tf.unstack(inputs, axis=-1)
        # print(len(ch_))
        # ch_ = tf.stack([self.__pad(_) for _ in ch_], axis=-1)
        # __ = self.__splitch_for_idwt(ch_)
        # __ = self.__splitch_for_idwt(inputs)
        # __ = [self.__rejoin(_) for _ in __]# axis=-1)

        # out = tf.stack([self.__idwt(_) for _ in __], axis=-1)

        ch_ = [self.__idwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        # print('+', h_[0].shape)
        out = tf.stack(ch_, axis=-1)
        # out = ch_
        # L = self.L
        return out#[:,L:-L,L:-L,:]

    @tf.function
    def __idwt(self, x_dwtcoeffs):
        # __ = x_dwtcoeffs
        # _r = tf.transpose(__, perm=[0,2,1])
        # _r = tf.cast(self.A.T, tf.float32) @ _r
        # ans = self.A.T@tf.transpose(_r,perm=[0,2,1])
        # # print('*', ans.shape)
        # # plt.imshow(ans[0,:,:]), plt.show()
        # return ans#[L:-L,L:-L,:]
        x_dwtcoeffs = tf.squeeze(x_dwtcoeffs, axis=-1)
        _LLLHHLHH = x_dwtcoeffs
        A = tf.cast(self.A, tf.float32)
        # print('>',x_dwtcoeffs.shape)
        
        _ = tf.transpose(_LLLHHLHH,[0,2,1])
        # print('>>', _.shape)
        _AT = tf.cast(tf.transpose(A,[1,0]), dtype=tf.float32)
        # print('>>',_AT.shape, _.shape)
        _r = tf.einsum('ij,bjk->bik', _AT, _)
        _rT = tf.transpose(_r,[0,2,1])
        _r = tf.einsum('ij,bjk->bik', _AT, _rT)
        # plt.imshow(_r)
        # _LLLHHLHH.dtype, _AT.dtype, _r.shape
        # print('+>',_r.shape)
        return _r
        
    
    # def __pad(self, x):
    #     paddings = tf.constant([[0,0], [self.pw, self.pw,], [self.pw, self.pw]])
    #     x_padded = tf.pad(x, paddings, self.padding_mode)
    #     # print('+',x_padded.shape) 
    #     return x_padded

    def __splitch_for_idwt(self,x):
        w = int(x.shape[-1]/4)
        __ = [tf.stack([x[:,:,:,k], x[:,:,:,k+1*w], x[:,:,:,k+2*w], x[:,:,:,k+3*w]],  axis=-1) for k in range(w)]
        return __
    #__ = __splitch_for_idwt(coeffs)

    def __rejoin(self,subbands):
        el = tf.unstack(subbands, axis=-1)
        _1 = tf.concat([el[0][:,:,:], el[2][:,:,:]], axis=-1)
        _1.shape
        _2 = tf.concat([el[1][:,:,:], el[3][:,:,:]], axis=-1)
        _1.shape
        _ = tf.concat([_1, _2], axis=-2)
        _.shape
        return _
    #tf.stack([rejoin(_) for _ in __], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width, new_channels = height *2, width *2, channels //4
    #     return (batch_size, new_height, new_width, new_channels)
   
    def get_config(self):
        config = super(IDWT2D, self).get_config()
        return config
    