"""ॐ
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

from VolterraSys.LSIVolterra1D import LSIVolterra1D
from VolterraSys.LSIVolterra2D import LSIVolterra2D
from VolterraSys.LSIVolterra3D import LSIVolterra3D
from VolterraSys.QSIVolterra1D import QSIVolterra1D
from VolterraSys.QSIVolterra2D import QSIVolterra2D
from VolterraSys.QSIVolterra3D import QSIVolterra3D

## MRA kernels
# default wave=haar
LSIVolterra1D(filters=1, kernel_size=4, wave='haar')
LSIVolterra2D(filters=1, kernel_size=4, wave='haar')
LSIVolterra3D(filters=1, kernel_size=4, wave='haar')

QSIVolterra1D(filters=1, kernel_size=4, wave='haar')
QSIVolterra2D(filters=1, kernel_size=4, wave='haar')
QSIVolterra3D(filters=1, kernel_size=4, wave='haar')



## Natural kenels
# The argument ```wave=None``` for natural domain computations
LSIVolterra1D(filters=1, kernel_size=4, wave=None)
LSIVolterra2D(filters=1, kernel_size=4, wave=None)
LSIVolterra3D(filters=1, kernel_size=4, wave=None)

QSIVolterra1D(filters=1, kernel_size=4, wave=None)
QSIVolterra2D(filters=1, kernel_size=4, wave=None)
QSIVolterra3D(filters=1, kernel_size=4, wave=None)