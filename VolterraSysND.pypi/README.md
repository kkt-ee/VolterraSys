# VolterraSys: Multidimensional linear and nonlinear Volterra kernels in natural and multiresolution bases

<!-- [![PyPI Version](https://img.shields.io/pypi/v/VolterraSys?label=PyPI&color=gold)](https://pypi.org/project/VolterraSys/)  -->
<!-- [![PyPI Version](https://img.shields.io/pypi/pyversions/VolterraSys)](https://pypi.org/project/VolterraSys/) -->
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15--2.19-darkorange)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-2--3-darkred)](https://keras.io/)
[![CUDA Version](https://img.shields.io/badge/cuda-12.5.1-green)](https://developer.nvidia.com/cuda-toolkit)
[![MIT](https://img.shields.io/badge/license-GPLv3-deepgreen.svg?style=flat)](https://github.com/kkt-ee/VolterraSys/LICENSE)



```python
## A shift variant kernel
from VolterraSys.LSVariantVolterra1D import LSVariantVolterra1D
```

```python
## Shift invariant LSI and QSI Volterra kernel
from VolterraSys.LSIVolterra1D import LSIVolterra1D
from VolterraSys.LSIVolterra2D import LSIVolterra2D
from VolterraSys.LSIVolterra3D import LSIVolterra3D
from VolterraSys.QSIVolterra1D import QSIVolterra1D
from VolterraSys.QSIVolterra2D import QSIVolterra2D
from VolterraSys.QSIVolterra3D import QSIVolterra3D
```

```python
# QSIVolterra1D(filters=1, kernel_size=4)
# QSIVolterraND(filters=1, kernel_size=4)
```



```python
## MRA kernels
# default wave=haar
LSIVolterra1D(filters=1, kernel_size=4, wave='haar')
LSIVolterra2D(filters=1, kernel_size=4, wave='haar')
LSIVolterra3D(filters=1, kernel_size=4, wave='haar')

QSIVolterra1D(filters=1, kernel_size=4, wave='haar')
QSIVolterra2D(filters=1, kernel_size=4, wave='haar')
QSIVolterra3D(filters=1, kernel_size=4, wave='haar')
```

The argument ```wave=None``` for natural domain computations



```python
## Natural kenels
# The argument ```wave=None``` for natural domain computations
LSIVolterra1D(filters=1, kernel_size=4, wave=None)
LSIVolterra2D(filters=1, kernel_size=4, wave=None)
LSIVolterra3D(filters=1, kernel_size=4, wave=None)

QSIVolterra1D(filters=1, kernel_size=4, wave=None)
QSIVolterra2D(filters=1, kernel_size=4, wave=None)
QSIVolterra3D(filters=1, kernel_size=4, wave=None)
```