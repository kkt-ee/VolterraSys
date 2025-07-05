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
from TFDWT3D.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT3D.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit
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
    
if __name__ == '__main__':

    # 1D dwt
    N = 256 # length of the sequence
    input_shape = (N,1)
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = '18'
    x = np.random.rand(input_shape[0])
    # x = tmpx
    plt.figure(figsize=(15,2))
    plt.plot(x, 'g.-',label='$x$')
    plt.legend(), plt.grid()
    plt.title('$x$')
    plt.show()
    print(f"Raw x shape {x.shape}")



    mother_wavelet = 'bior3.1' # max 'db8' for lenght 16 signal
    # mother_wavelet = 'haar' # max 'db8' for lenght 16 signal
    # mother_wavelet = 'db10' # max 'db8' for lenght 16 signal
    newx = tf.expand_dims(tf.expand_dims(x,-1),0)
    print(f'x shape {newx.shape}')
    dwtout = DWT1D(wave=mother_wavelet)(newx)
    print(f'DWT(x) shape {dwtout.shape}, \nDWT(x) := {dwtout}')

    # print(dwtout.shape)
    idwtout = IDWT1D(wave=mother_wavelet)(dwtout)
    print(f'IDWT(DWT(x)) shape {idwtout.shape} \nIDWT(DWT(x)) := {idwtout}')

    print(f'Check perfect reconstruction \nIDWT(DWT(x)) := {idwtout.numpy()} \n\nInput x is {x} ')


    

    plt.figure(figsize=(16,2))
    plt.plot(x,'o-')
    plt.plot(idwtout.numpy()[0,:], '.--'), plt.title(f"reconstuction using {mother_wavelet}")







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
from TFDWT3D.GETDWTFiltersOrtho import GETDWTFiltersOrtho
from TFDWT3D.get_A_matrix_dwt_analysisFB_unit import get_A_matrix_dwt_analysisFB_unit
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
    
if __name__=='__main__':
    # import pywt.data
    # Load image
    # x1 = pywt.data.camera()

    import cv2
    x = cv2.imread(f'/home/kishoretarafdar/src/lena.jpg',cv2.IMREAD_GRAYSCALE) #test.jpg
    print('raw x shape:', x.shape)
    x = cv2.resize(x, (512,512))
    print('x shape:', x.shape)
    #x = x/np.max(x)
    plt.imshow(x,label='$x$')
    plt.title('input $x$')

    x.shape
    xnew = tf.expand_dims(tf.expand_dims(x, axis=-1), axis=0)
    xnew.shape

    x1=x
    xnew1 = tf.expand_dims(tf.expand_dims(x1, axis=-1), axis=0)
    xnew1.shape
    # del x, x1

    xnew = tf.cast(xnew, dtype=tf.float32)
    xnew1 = tf.cast(xnew1, dtype=tf.float32)
    _1 = tf.concat([xnew,xnew1,xnew],axis=-1)
    _2 = tf.concat([xnew1,xnew,xnew1],axis=-1)
    xnew = tf.cast(tf.concat([_1,_2], axis=0), dtype=tf.float32)
    _1.shape,_2.shape, xnew.shape, _1.dtype,_2.dtype, xnew.dtype
    del _1, _2



    wave = 'haar'
    wave = 'db6'
    wave = 'bior3.1'
    coeffs = DWT2D(wave=wave)(xnew)
    print(coeffs.shape)
    import matplotlib.pyplot as plt
    _ = coeffs
    for b in range(_.shape[0]):
        plt.figure(figsize=(16,8))
        for c in range(_.shape[-1]):
            plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
        plt.show()    

    out = IDWT2D(wave=wave)(coeffs)
    # [_.shape for _ in out]
    print(out.shape, out.dtype)
    import matplotlib.pyplot as plt
    _ = out
    for b in range(_.shape[0]):
        plt.figure(figsize=(12,4))
        for c in range(_.shape[-1]):
            plt.subplot(1, _.shape[-1], c+1), plt.imshow(_[b,:,:,c]), plt.title(f'bat.{b}, ch.{c}')
        plt.show()
            






























#%%
import tensorflow as tf
from TFDWT3D.DWTIDWT1Dtfv1 import DWT1D, IDWT1D


class DWTlv1VolterraKernel1Dio(tf.keras.layers.Layer):
    """Multiresolution Volterra Kernel

        Sequence input x[n]

        DWT Lv 1 basis for both 



       --@KKT, 7-Feb-2025
    """
    def __init__(self, filter_size, Ψ='db4', **kwargs):
        super(DWTlv1VolterraKernel1Dio, self).__init__(**kwargs)
        self.filter_size = kernel_size  # Tuple (kernel_height, kernel_width)
        # self.out_channels = out_channels
        self.Ψ = Ψ 

    def build(self, input_shape):
        N = input_shape[1]
        in_channels = input_shape[-1]
      
        # Initialize vectors for height and width directions
        self.h = self.add_weight(
            name='kernel_h',
            shape=(self.filter_size, in_channels),#, self.out_channels),
            initializer='glorot_uniform',
            trainable=True
        )

        # Precompute circular indices
        n = tf.range(N)
        self.row_indices = tf.math.mod(n[:, tf.newaxis] - n, self.N)

        # """support for orthonormal basis for kernel"""
        # self.ϕ_l = self.add_weight(
        #     name='kernel_ϕ_l',
        #     shape=(N//2, int(in_channels*2)),
        #     initializer='glorot_uniform',
        #     trainable=True
        # )

        # """support for orthonormal basis for y[n]"""
        # self.Φ_v = [
        #     self.add_weight(
        #         name=f'kernel_ϕ_{v}',
        #         shape=(N//2, int(in_channels*2)),
        #         initializer='glorot_uniform',
        #         trainable=True)
        #         for v in range(N//2)
        #         ]

        # self.TraceConv1Dio = TraceConv1Dio(kernel_size=self.kernel_size)
        super(DWTlv1VolterraKernel1Dio, self).build(input_shape)

    def call(self, x):
        ## DWT(x) \times DWT(x)
        self.α = DWT1D(wave=self.Ψ)(x) # batch, N/2, subbands
        
        Npoint = x.shape[1]
        paddings = [
            [0, Npoint - self.filter_size],  # Depth dimension
            [0, 0],
        ]
        h_padded = tf.pad(self.h, paddings, mode='CONSTANT', constant_values=0)
        # h_padded = tf.transpose(h_padded,perm=[4,3,0,1,2])
        
        # batch_size = tf.shape(x)[0]
        # batch_indices = tf.expand_dims(self.row_indices, 0)  # (1, N)
        # print('bi+',batch_indices.shape)
        # batch_indices = tf.tile(batch_indices, [batch_size, 1])  # (batch, N)
        # print('bi+',batch_indices.shape)

        # # Gather shifted rows and columns
        # rows_shifted = tf.gather(x, self.row_indices, axis=1)  # (batch, N, N, N, ch)

        
        
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
            'filter_size': self.filter_size,
        })
        return config

x = tf.random.normal([10, 16, 12])
x = tf.random.normal([1, 16, 1])
# x = tf.random.normal([1, 3, 1])
y = DWTlv1VolterraKernel1Dio(kernel_size=5, Ψ='haar')(x)
y, y.shape
## reduce kernel size than signal size # DONE
## now check with determined inputs












#%% multilevel DWT
# mid = N//2
lh1 = DWT1D(wave)(x)
lh2 = DWT1D(wave)(lh1[:,:N//2,:])
lh2, lh1[:,N//2:,:]
ix = IDWT1D()(tf.concat([IDWT1D()(lh2),lh1[:,N//2:,:]], axis=1))
x, ix, tf.cast(x, dtype=tf.float32)-ix


lh3 = DWT1D(wave)(lh2[:,:N//4,:])
lh3, lh2[:,:N//4,:], lh1[:,N//2:,:]
IDWT1D()(tf.concat([IDWT1D()(tf.concat([IDWT1D()(lh3), lh2[:,N//4:,:]], axis=1)), lh1[:,:N//2,:]], axis=1)), x
# lh3.shape, lh3[:,:N//8,:].shape
# lh4 = DWT1D(wave)(lh3[:,:N//8,:])
# lh3.shape, 
# lh4.shape
# lh4, lh3[:,N//8:,:], lh2[:,:N//4,:], lh1[:,N//2:,:]
# IDWT1D()(tf.concat([IDWT1D()(tf.concat([IDWT1D()(tf.concat([IDWT1D()(lh4), lh3[:,N//8:,:]], axis=1)), lh2[:,:N//4,:]], axis=1)), lh1[:,N//2:,:]], axis=1)), x
# IDWT1D()(tf.concat([il1,h1],axis=1)), x
# IDWT1D()(tf.concat([il1,h1],axis=1))
# l1.shape, h1.shape, N
# lh1, IDWT1D()(tf.concat([lh1[:,N//4:,:],lh2],axis=1))

# lh3 = DWT1D(wave)(lh2[:,:N//8,:])
# lh1.shape, lh2.shape
# _ = tf.concat([lh2,lh1[:,N//4:,:]],axis=1)
# ilh1 = IDWT1D(wave)(_)

# lh1, ilh1

# tf.concat(lh3,lh2[:,:N//8,:],axis=1).shape
# lh2 = DWT1D(wave)()



























#%% Linear kernel for sequence input and output ym, m=1
# x = [1,2,3,4,5,6,7,8]
x = [1,0,0,0,0,0,0,0]
x = [0,1,0,0,0,0,0,0]
x = [0,0,1,0,0,0,0,0]
x = [0,0,0,1,0,0,0,0]
# x = [0,0,0,0,1,0,0,0]
# x = [0,0,0,0,0,1,0,0]
# x = [0,0,0,0,0,0,1,0]
# x = [0,0,0,0,0,0,0,1]
# x = [0,0,0,1]
# x = [1,0,1,0,1,0,1,0]
# x = [1,1,1,1,0,0,0,0]
# x = [1,2,3,1]
# x = [0,0,0,0,1,1,1,1]
wave = 'haar'
# wave = 'db2'
# wave = 'db3'

## filter init
h = tf.constant([1,2,1,0])
## Matching filter size with input size with zero padding
L = h.shape[0]

# row_indices = tf.math.mod(n[:, tf.newaxis] - n, L)
Npoint = len(x)
paddings = [
           [0, Npoint - L],  # Depth dimension
        ]
h_padded = tf.pad(h, paddings, mode='CONSTANT', constant_values=0)
h_padded.shape


## Creating x[n-k] for all n
N = Npoint
n = tf.range(N)
row_indices = tf.math.mod(n[:, tf.newaxis] - n, N)
row_indices
h_rows_shifted = tf.gather(h_padded, row_indices, axis=0)
print(h_rows_shifted.shape)
_ = tf.transpose(tf.expand_dims(tf.cast(h_rows_shifted, dtype=tf.float32), axis=0), perm=[0,2,1])
h_init_T = tf.transpose(h_init, perm=[0,2,1])
H = DWT1D(wave)(h_init_T)
print('H1 shape ',H.shape)
## kernel ready

# prepare x 
x = tf.constant(x)
x
x = tf.expand_dims(tf.expand_dims(x, axis=0),axis=-1)
x.shape
α = DWT1D(wave)(tf.cast(x, dtype=tf.float32))
α.shape

## convolution like in wavelet domain
β = tf.einsum('bic,ij->bjc',α,tf.squeeze(H))
β.shape
β
y1 = IDWT1D(wave)(β)
y1.shape,y1

# alpha1 = α
# beta1 = β
# alpha1, beta1 
# import matplotlib.pyplot as plt
# plt.figure(figsize=(2,4))
# plt.subplot(2,1,1), plt.stem(tf.squeeze(x).numpy(),'k')
# plt.subplot(2,1,2), plt.stem(tf.squeeze(y1).numpy(),'k')
# plt.tight_layout()
# tf.squeeze(x).numpy().shape, tf.squeeze(y1).numpy().shape




#%% verify
x
rows_shifted[1,:]
tf.squeeze(x).shape
tf.einsum('i,ij->j', tf.squeeze(x),rows_shifted), x, rows_shifted
#%%
tf.reduce_sum(tf.squeeze(x)* rows_shifted[0,:])
# rows_shifted.shape




# %% Quadratic kernel (m=2)
"""Quadratic kernel"""
# x = [0,0,0,1,0,0,0,0]
# filter init
# h = tf.constant([1, 2, 1])
h2 = np.einsum('i,j->ij', h, h)
print(h2)
L = h.shape[0] #filter length
# n = tf.range(L) 
Npoint = x.shape[1]
# Npoint = len(x)
paddings = [
           [0, Npoint - L],  # Depth dimension
           [0, Npoint - L]
        ]
# x
Npoint, paddings
h_padded = tf.pad(h2, paddings, mode='CONSTANT', constant_values=0)
print(h_padded)
h_padded.shape

# _ = tf.expand_dims(tf.expand_dims(h_padded, axis=0),axis=-1)
# _.shape
# from TFDWT3D.DWTIDWT2Dtfv1 import DWT2D, IDWT2D
# h_init = IDWT2D()(tf.cast(_, dtype=tf.float32))
# h_init


## creating h[n-k1,n-k2]
N = Npoint
n = tf.range(N)
# Create circular indices for all possible shifts
# row_indices = tf.math.mod(n[:, tf.newaxis] - tf.range(N), L)  # [N, N]
row_indices = tf.math.mod(n[:, tf.newaxis] - n, N)
rows_shifted = tf.gather(h_padded, row_indices, axis=0)         # [N, N, N]
rows_shifted
columns_shifted = tf.gather(rows_shifted, row_indices,   # [N, N, N]
                                axis=2, batch_dims=1)
columns_shifted, columns_shifted.shape

h_init = IDWT2D()(tf.expand_dims(tf.cast(columns_shifted, dtype=tf.float32), axis=0))
h_init
## remember from NLSI theory of Volterra kernels and compute below
h_init = tf.concat([tf.expand_dims(tf.einsum('ijkl->il',h_init[:,:n,:n,:]), axis=1) for n in range(h_init.shape[1])], axis=1)
H2 = DWT1D()(h_init)
H2, H2.shape


# x = tf.constant(x)
# x
# x = tf.expand_dims(tf.expand_dims(x, axis=0),axis=-1)
# x.shape
α = DWT1D(wave)(tf.cast(x, dtype=tf.float32))
α.shape

## convolution like in wavelet domain
β = tf.einsum('bic,ij->bjc',α,tf.squeeze(H2))
β.shape
β
y2 = IDWT1D(wave)(β)
y2.shape,y1

# #%%
# alpha2 = α
# beta2 = β
# alpha2, beta2 
plt.figure(figsize=(2,5))
plt.subplot(3,1,1), plt.stem(tf.squeeze(x).numpy(),'k', basefmt="k"), plt.title(f'w. {wave}')
plt.subplot(3,1,2), plt.stem(tf.squeeze(y1).numpy(),'k', basefmt="k")
plt.subplot(3,1,3), plt.stem(tf.squeeze(y2).numpy(),'k', basefmt="k")
plt.tight_layout()
# tf.squeeze(x).numpy().shape, tf.squeeze(y).numpy().shape


































#%% Linear kernel for image input
import numpy as np
# x = [1,2,3,4,5,6,7,8]
# x = [1,0,0,0,0,0,0,0]
# x = [0,1,0,0,0,0,0,0]
# x = [0,0,1,0,0,0,0,0]
# x = [0,0,0,1,0,0,0,0]
# x = [0,0,0,0,1,0,0,0]
# x = [0,0,0,0,0,1,0,0]
# x = [0,0,0,0,0,0,1,0]
# x = [0,0,0,0,0,0,0,1]
# x = [0,0,0,1]
# x = [0,1,0,1,0,1,0,1]
x = np.array([[1,0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]])
# x = np.array([[0,1,0,0,0,0,0,0], 
#               [0,0,0,0,0,0,0,0], 
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0]])
# x = np.array([[0,0,0,0,0,0,0,0], 
#               [1,0,0,0,0,0,0,0], 
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0]])              

wave = 'haar'
wave = 'db2'

h1 = np.array([[1, 2, 1], 
              [2, 4, 2], 
              [1, 2, 1]])

# ## filter init
# h1 = np.array([[1, 1, -1], 
#               [1, 2, -1], 
#               [1, -1, -1]])

# h1 = np.array([[1, 1], 
#               [1, 21]])
## Matching filter size with input size with zero padding
L = h2.shape[0]

# row_indices = tf.math.mod(n[:, tf.newaxis] - n, L)
Npoint = x.shape[0]
paddings = [
           [0, Npoint - L],  # height dimension
           [0, Npoint - L],  # width dimension
        ]
h1_padded = tf.pad(h1, paddings, mode='CONSTANT', constant_values=0)
h1_padded.shape




# import tensorflow as tf

# Example input
# N = 8  # Size of the tensor
# h = tf.reshape(tf.range(N * N, dtype=tf.float32), (N, N))  # Example tensor of shape (N, N)

# Create a meshgrid for n1, n2, k1, k2
n1 = tf.range(Npoint)  # [0, 1, ..., N-1]
n2 = tf.range(Npoint)  # [0, 1, ..., N-1]
k1 = tf.range(Npoint)  # [0, 1, ..., N-1]
k2 = tf.range(Npoint)  # [0, 1, ..., N-1]

# Create a grid of indices for n1, n2, k1, k2
n1, n2, k1, k2 = tf.meshgrid(n1, n2, k1, k2, indexing='ij')  # Shape (N, N, N, N)

# Compute circularly shifted indices
shifted_n1 = tf.math.mod(n1 - k1, Npoint)  # Circular shift for n1
shifted_n2 = tf.math.mod(n2 - k2, Npoint)  # Circular shift for n2

# Gather the values from h using the shifted indices
h_shifted = tf.gather_nd(h1_padded, tf.stack([shifted_n1, shifted_n2], axis=-1))

# Result
print("Original h:")
print(h.numpy())
print("\nShape of h_shifted:", h_shifted.shape)
print("\nShifted h (first few elements):")
print(h_shifted[0, 0, 0:3, 0:3].numpy())  # Example slice for visualization
h_shifted.shape, h_shifted[5,4,:,:]
# h_shifted
h_shifted = tf.transpose(h_shifted,perm=[0,2,3,1])  # n1,:,:,n2
h_init = IDWT2D(wave)(tf.cast(h_shifted, dtype=tf.float32))
H = DWT2D(wave)(h_init)
n1,n2 = 6,5
print('H1 shape ',H.shape)
h_shifted[n1,:,:,n2], h_init[n1,:,:,n2], 
H[n1,:,:,n2]

# prepare x 
x = tf.constant(x)
x
x = tf.expand_dims(tf.expand_dims(x, axis=0),axis=-1)
x.shape

α = DWT2D(wave)(tf.cast(x, dtype=tf.float32))
α.shape

## convolution like in wavelet domain
β = tf.einsum('bijc,uijv->buvc',α,H)

# β = tf.einsum('bic,ij->bjc',α,H)
β.shape
β
y = IDWT2D(wave)(β)
y.shape,y[0,:,:,0]

# alpha1 = α
# beta1 = β
# alpha1, beta1 


plt.figure(figsize=(2,3))
plt.subplot(2,1,1), plt.imshow(tf.squeeze(x).numpy()), plt.title('x')
plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2), plt.imshow(tf.squeeze(y).numpy()),  plt.title(f'y ({wave})')
plt.xticks([]), plt.yticks([])
plt.tight_layout()







#%%
h_ = h/tf.reduce_max(y)
y_ = y/tf.reduce_max(y)
h_, y_
#%%
_ = h/tf.squeeze(y)
_
#%%
# y.shape
tf.squeeze(tf.squeeze(y)[0,1])
print(f"{h[0,0]}/{tf.squeeze(y)[0,0]}={h[0,0]/tf.squeeze(y)[0,0]}")
print(f"{h[0,1]}/{tf.squeeze(y)[0,1]}={h[0,1]/tf.squeeze(y)[0,1]}")
#%%
h[0]
_ = tf.where(tf.math.is_inf(_), 0, _)
_ = tf.where(tf.math.is_nan(_), 0, _)
# 1-_
_






























#%% Linear kernel for vol input incomplete!!
import numpy as np
# x = [1,2,3,4,5,6,7,8]
x = [1,0,0,0,0,0,0,0]
x = [0,1,0,0,0,0,0,0]
x = [0,0,1,0,0,0,0,0]
x = [0,0,0,1,0,0,0,0]
x = [0,0,0,0,1,0,0,0]
# x = [0,0,0,0,0,1,0,0]
# x = [0,0,0,0,0,0,1,0]
# x = [0,0,0,0,0,0,0,1]
# x = [0,0,0,1]
# x = [0,1,0,1,0,1,0,1]
x = np.array([[0,0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0,0], 
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,1]])
# x = np.array([[0,0,0,0,0,0,0,0], 
#               [0,1,0,0,0,0,0,0], 
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0]])

wave = 'haar'
# wave = 'db2'

## filter init
h1 = np.array([[1, 1, -1], 
              [1, 2, -1], 
              [1, -1, -1]])

h1 = np.array([[1, 1], 
              [1, 21]])
## Matching filter size with input size with zero padding
L = h2.shape[0]

# row_indices = tf.math.mod(n[:, tf.newaxis] - n, L)
Npoint = x.shape[0]
paddings = [
           [0, Npoint - L],  # height dimension
           [0, Npoint - L],  # width dimension
           [0, Npoint - L],  # width dimension
        ]
h1_padded = tf.pad(h1, paddings, mode='CONSTANT', constant_values=0)
h1_padded.shape




# import tensorflow as tf

# Example input
# N = 8  # Size of the tensor
# h = tf.reshape(tf.range(N * N, dtype=tf.float32), (N, N))  # Example tensor of shape (N, N)

# Create a meshgrid for n1, n2, k1, k2
n1 = tf.range(Npoint)  # [0, 1, ..., N-1]
n2 = tf.range(Npoint)  # [0, 1, ..., N-1]
n3 = tf.range(Npoint)  # [0, 1, ..., N-1]
k1 = tf.range(Npoint)  # [0, 1, ..., N-1]
k2 = tf.range(Npoint)  # [0, 1, ..., N-1]
k3 = tf.range(Npoint)  # [0, 1, ..., N-1]


# Create a grid of indices for n1, n2, k1, k2
n1, n2, n3, k1, k2, k3 = tf.meshgrid(n1, n2, n3, k1, k2, k3, indexing='ijk')  # Shape (N, N, N, N, N, N)

# Compute circularly shifted indices
shifted_n1 = tf.math.mod(n1 - k1, Npoint)  # Circular shift for n1
shifted_n2 = tf.math.mod(n2 - k2, Npoint)  # Circular shift for n2
shifted_n3 = tf.math.mod(n3 - k3, Npoint)  # Circular shift for n2

# Gather the values from h using the shifted indices
h_shifted = tf.gather_nd(h1_padded, tf.stack([shifted_n1, shifted_n2, shifted_n3], axis=-1))

# Result
print("Original h:")
print(h.numpy())
print("\nShape of h_shifted:", h_shifted.shape)
print("\nShifted h (first few elements):")
print(h_shifted[0, 0, 0, 0:3, 0:3, 0:3].numpy())  # Example slice for visualization
h_shifted.shape, h_shifted[5,4,1,:,:,:]
# h_shifted
h_shifted = tf.transpose(h_shifted,perm=[0,2,3,1])  # n1,:,:,n2
h_init = IDWT2D(wave)(tf.cast(h_shifted, dtype=tf.float32))
H = DWT2D(wave)(h_init)
n1,n2 = 6,5
print('H1 shape ',H.shape)
h_shifted[n1,:,:,n2], h_init[n1,:,:,n2], 
H[n1,:,:,n2]

# prepare x 
x = tf.constant(x)
x
x = tf.expand_dims(tf.expand_dims(x, axis=0),axis=-1)
x.shape

α = DWT2D(wave)(tf.cast(x, dtype=tf.float32))
α.shape

## convolution like in wavelet domain
β = tf.einsum('bijc,uijv->buvc',α,H)

# β = tf.einsum('bic,ij->bjc',α,H)
β.shape
β
y = IDWT2D(wave)(β)
y.shape,y[0,:,:,0]

# alpha1 = α
# beta1 = β
# alpha1, beta1 






