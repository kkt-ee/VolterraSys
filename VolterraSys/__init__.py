"""ॐ
    VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2025 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""


"""
## A linear kernel (shift variant)
from VolterraSys.LinearVolterra1D import LinearVolterra1D


## Linear (m=1) shift invariant (LSI) Volterra kernel layers in wavelet and natural bases
from VolterraSys.LSIVolterra1D import LSIVolterra1D
from VolterraSys.LSIVolterra2D import LSIVolterra2D
from VolterraSys.LSIVolterra3D import LSIVolterra3D

# Quadratic (m=2) shift invariant (QSI) Volterra kernel layers in wavelet and natural bases
from VolterraSys.QSIVolterra1D import QSIVolterra1D
from VolterraSys.QSIVolterra2D import QSIVolterra2D
from VolterraSys.QSIVolterra3D import QSIVolterra3D

## Multiresolution or orthogonal and biorthogonal wavelet domain kernel layers
# default wave='haar' and supports wavelet families: 'db', 'sym', 'coif', 'bior', 'rbio'  
LSIVolterra1D(filters=1, kernel_size=4, wave='haar')
LSIVolterra2D(filters=1, kernel_size=4, wave='haar')
LSIVolterra3D(filters=1, kernel_size=4, wave='haar')

QSIVolterra1D(filters=1, kernel_size=4, wave='haar')
QSIVolterra2D(filters=1, kernel_size=4, wave='haar')
QSIVolterra3D(filters=1, kernel_size=4, wave='haar')



## Natural domain kernels
# The argument ```wave=None``` for natural domain computations
LSIVolterra1D(filters=1, kernel_size=4, wave=None)
LSIVolterra2D(filters=1, kernel_size=4, wave=None)
LSIVolterra3D(filters=1, kernel_size=4, wave=None)

QSIVolterra1D(filters=1, kernel_size=4, wave=None)
QSIVolterra2D(filters=1, kernel_size=4, wave=None)
QSIVolterra3D(filters=1, kernel_size=4, wave=None)
"""

__version__="0.0.1"
