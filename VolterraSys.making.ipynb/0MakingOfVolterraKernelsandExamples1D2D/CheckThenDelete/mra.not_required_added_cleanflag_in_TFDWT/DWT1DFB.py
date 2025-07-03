#%%
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
#import pywt
import numpy as np
# print(f"\
#     \n tf {tf.__version__}, \
#     \n matplotlib {matplotlib.__version__} \
#     \n scipy {scipy.__version__}\
#     \n pywavelets {pywt.__version__} (for comparing tf results)")

#### OKAY
# def get_A_matrix_dwt_analysisFB_unit(h0,h1,N:int):
#     """returns Analysis Matrix A"""
#     L = len(h0)
#     __H_branch_row = lambda h0,N: np.concatenate((h0, np.zeros(int(N - len(h0)))))
#     __H_start_row = lambda _row: np.roll(_row, shift=-L+2, axis=None)
#     __H_start_row(__H_branch_row(h0,N))
#     __H_branch= lambda _row: [np.roll(_row, shift=k*2, axis=None) for k in range(_row.shape[0]//2)]#-len(h0)+1)]
#     H0 = __H_branch(__H_start_row(__H_branch_row(h0,N)))
#     H1 = __H_branch(__H_start_row(__H_branch_row(h1,N)))
#     # return np.concatenate((H0[:int(len(H0)/2)], H1[:int(len(H1)/2)]))
#     return np.concatenate((H0, H1))

# A = get_A_matrix_dwt_analysisFB_unit(h0,h1,N)
# A.shape, A


## OK
# import pywt
from TFDWT.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit
@keras.saving.register_keras_serializable()
class DWT1D(tf.keras.layers.Layer):
    """1D DWT layer @kkt"""
    def __init__(self, wave='haar', **kwargs):
        super(DWT1D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        self.h0, self.h1 = w.analysis()
        self.L = len(self.h0)
        # self.padding_mode = 'symmetric' #reflect'
        # self.N
        # self.A
        
    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = input_shape[1]
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # self.pw = (self.A.shape[0]-self.N)//2
        super(DWT1D, self).build(input_shape) 
       
    
    def call(self, inputs):
        # ch_ = [self.__pad(inputs[..., i:i+1]) for i in range(self.num_channels)]
        # ch_ = tf.concat(ch_, axis=-1)
        # ch_ = [self.__dwt(ch_[..., i:i+1]) for i in range(self.num_channels)]
        ch_ = [self.__dwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.concat(ch_, axis=-1)
        #
        # out = self.__extract_2subbands(out)
        return out

    # @tf.function
    def __dwt(self, x):
        # print('>>',x.shape)
        x = tf.squeeze(x, axis=-1)
        dwtout = tf.cast(self.A, tf.float32)@tf.transpose(tf.cast(x, tf.float32), perm=[1,0])
        #OR einsum !!
        # dwtout = tf.einsum('ij,bjk->bik', tf.cast(self.A, tf.float32), tf.transpose(tf.cast(x, tf.float32), perm=[1,0]))
        dwtout = tf.transpose(dwtout,perm=[1,0])
        # print('++',dwtout.shape)
        return tf.expand_dims(dwtout, axis=-1)

    # def __pad(self, x):
    #     x = tf.squeeze(x, axis=-1)
    #     # paddings = tf.constant([[0,0], [self.pw, self.pw,], [self.pw, self.pw]])
    #     paddings = tf.constant([[0,0], [self.pw, self.pw]])

    #     x_padded = tf.pad(x, paddings, self.padding_mode)
    #     print('+',x_padded.shape) 
    #     # return x_padded
    #     return tf.expand_dims(x_padded, axis=-1)

    def __extract_2subbands(self,LH_padded):
        """returns 2 subbands L, H from DWT Analysis bank o/p --@k"""
     
        _mid = int(LH_padded.shape[1]/2)
        L = LH_padded[:,:_mid,:]
        H = LH_padded[:,_mid:,:]
        
        # _i = int((L.shape[1] - self.N/2)/2)
        # out = tf.concat([L[:,_i:-_i,:], H[:,_i:-_i,:]], axis=-1)
        out = tf.concat([L, H], axis=-1)
        return out

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width = height // 2, width // 2
    #     return (batch_size, new_height, new_width, 4 * channels)
    
    def get_config(self):
        config = super(DWT1D, self).get_config()
        return config



# ## WORKING IDWT similar to wavetf
# import pywt
# from GETDWTFiltersOrtho import GETDWTFiltersOrtho
@keras.saving.register_keras_serializable()
class IDWT1D(tf.keras.layers.Layer):
    """1D IDWT layer @kkt"""
    def __init__(self, wave='haar', **kwargs):
        super(IDWT1D, self).__init__(**kwargs)
        w = GETDWTFiltersOrtho(wave)
        
        # self.biortho_flag = 0
        if 'bior' in wave or 'rbio' in wave:
            """BIORTHOGONAL wavelets"""
            # self.g0, self.g1 = w.synthesis()
            # self.h0, self.h1 = self.g0, self.g1
            # del self.g0, self.g1
            self.h0, self.h1 = w.synthesis()
            # self.biortho_flag = 1
            print(f"Biothogonal wavelet {wave}")
        else:
            """ORTHOGONAL wavelets"""
            self.h0, self.h1 = w.analysis()
        
        # g0, g1 = w.synthesis()
        self.L = len(self.h0)
        # self.padding_mode = 'symmetric'#'reflect'#

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.N = int(input_shape[1])#*2)
        self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # if self.biortho_flag:
            # self.S = get_A_matrix_dwt_analysisFB_unit(self.g0,self.g1,self.N)
        # self.pw = int(self.L/2)
        super(IDWT1D, self).build(input_shape) 

    def call(self, inputs):
        # self.N = int(inputs.shape[1]*2)
        # self.A = get_A_matrix_dwt_analysisFB_unit(self.h0,self.h1,self.N)
        # input_shape = tf.shape(inputs)

        #all input channels
        # ch_ = tf.unstack(inputs, axis=-1)
        # print(len(ch_))
        # ch_ = tf.stack([self.__pad(_) for _ in ch_], axis=-1)
        # __ = self.__splitch_for_idwt(ch_)
        # print('+', self.__idwt(inputs[..., 0:0+1]).shape)
        # __ = self.__splitch_for_idwt(inputs)
        # print('++', __[0].shape)
        # __ = [self.__rejoin(_) for _ in __]# axis=-1)
        # print('++', __[0].shape)
        # out = tf.stack([self.__idwt(_) for _ in __], axis=-1)

        ch_ = [self.__idwt(inputs[..., i:i+1]) for i in range(self.num_channels)]
        out = tf.stack(ch_, axis=-1)
        # print('+++', out.shape)
        # out = ch_
        # L = self.L
        return out#[:,L:-L,:]

    @tf.function
    def __idwt(self, x_dwtcoeffs):
        __ = x_dwtcoeffs
        # print('++', type(__))
        __ = tf.squeeze(__, axis=-1)
        # print('+++', __[0].shape)
        # perm=[1,0]
        # print('+++', __.shape)
        # if self.biortho_flag:
        #     """BIORTHOGONAL wavelet"""
        #     idwtout = tf.cast(self.A.T, tf.float32) @ tf.transpose(__,perm=[1,0])
        #     # idwtout = tf.transpose(idwtout,perm=[1,0])
        # else:
        #     """ORTHOGONAL wavelet"""
        idwtout = tf.cast(self.A.T, tf.float32) @ tf.transpose(__,perm=[1,0])
        idwtout = tf.transpose(idwtout,perm=[1,0])
        
        # idwtout = tf.expand_dims(idwtout)
      
        return idwtout#[L:-L,L:-L,:]
        
    
    # def __pad(self, x):
    #     paddings = tf.constant([[0,0], [self.pw, self.pw,]])
    #     x_padded = tf.pad(x, paddings, self.padding_mode)
    #     # print('+',x_padded.shape) 
    #     return x_padded

    def __splitch_for_idwt(self,x):
        w = int(x.shape[-1]/2)
        __ = [tf.stack([x[:,:,k], x[:,:,k+1*w]],  axis=-1) for k in range(w)]
        return __
    #__ = __splitch_for_idwt(coeffs)

    def __rejoin(self,subbands):
        el = tf.unstack(subbands, axis=-1)
        # print('++++', el)
        _1 = tf.concat([el[0][:,:], el[1][:,:]], axis=-1)
        # print('+++++', _1.shape)
        # _1.shape
        # _2 = tf.concat([el[1][:,:,:], el[3][:,:,:]], axis=-1)
        # _1.shape
        # _ = tf.concat([_1, _2], axis=-2)
        # _.shape
        return _1
    #tf.stack([rejoin(_) for _ in __], axis=-1)

    # def compute_output_shape(self, input_shape):
    #     batch_size, height, width, channels = input_shape
    #     new_height, new_width, new_channels = height *2, width *2, channels //4
    #     return (batch_size, new_height, new_width, new_channels)
   
    def get_config(self):
        config = super(IDWT1D, self).get_config()
        return config