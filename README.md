# VolterraSys: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases

[![PyPI Version](https://img.shields.io/pypi/v/VolterraSys?label=PyPI&color=gold)](https://pypi.org/project/VolterraSys/)
[![Python Versions](https://img.shields.io/pypi/pyversions/VolterraSys)](https://pypi.org/project/VolterraSys/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-required-darkorange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-deepgreen.svg?style=flat)](https://github.com/kkt-ee/VolterraSys/LICENSE)

`VolterraSys` provides TensorFlow/Keras layers for trainable multidimensional linear and quadratic Volterra kernels in wavelet and natural bases.

Copyright 2025 Kishore Kumar Tarafdar.
Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE).

## Capabilities

- linear Volterra kernel layer for 1D data (shift variant): `linearVolterra1D`.
- Linear shift invariant wavelet and natural basis Volterra kernel layers for 1D, 2D, 3D data: `LSIVolterra1D`, `LSIVolterra2D`, `LSIVolterra3D`.
- Quadratic shift invariant wavelet and natural basis Volterra kernel layers for 1D, 2D, 3D data: `QSIVolterra1D`, `QSIVolterra2D`, `QSIVolterra3D`.
- Multiresolution Volterra kernels with orthogonal wavelets such as `wave="haar"` (default) and biorthogonal wavelets `wave="bior1.3"`. Supports wavelet families: "db", "sym", "coif", "bior", "rbio"  
- Natural-domain kernels have `wave=None`.

## Dependencies

- TensorFlow (>=2.15)
- Inputs are TensorFlow tensors with channel-last layout.
- Wavelet domain requires the `TFDWT` package which can be installed via `pip install TFDWT`.
<!-- - Natural-domain computation uses `wave=None`; wavelet-domain computation uses a supported wavelet name. -->

## Installation

```bash
pip install VolterraSys
```

## Minimal Example

```python
import tensorflow as tf

# Linear shift invariant wavelet bases Volterra kernels
from VolterraSys.LSIVolterra1D import LSIVolterra1D
from VolterraSys.LSIVolterra2D import LSIVolterra2D
from VolterraSys.LSIVolterra3D import LSIVolterra3D

# Quadratic shift invariant wavelet bases Volterra kernels
from VolterraSys.QSIVolterra1D import QSIVolterra1D
from VolterraSys.QSIVolterra2D import QSIVolterra2D
from VolterraSys.QSIVolterra3D import QSIVolterra3D

# Natural-domain linear and quadratic kernels layer examples
# Linear
x1d = tf.random.normal([1, 32, 1])
y1d = LSIVolterra1D(filters=2, kernel_size=3, wave=None)(x1d)

# Quadratic
x2d = tf.random.normal([1, 8, 8, 1])
yq2d = QSIVolterra2D(filters=2, kernel_size=2, wave=None)(x2d)

x3d = tf.random.normal([1, 6, 6, 6, 1])
yq3d = QSIVolterra3D(filters=2, kernel_size=2, wave=None)(x3d)

print(y1d.shape, yq2d.shape, yq3d.shape)
```

## Wavelet domain kernel layer examples

```python
# Linear layers
layer1d = LSIVolterra1D(filters=1, kernel_size=4, wave="haar")
layer2d = LSIVolterra2D(filters=1, kernel_size=4, wave="haar")
layer3d = LSIVolterra3D(filters=1, kernel_size=4, wave="haar")

# Quadratic layers
qlayer1d = QSIVolterra1D(filters=1, kernel_size=4, wave='haar')
qlayer2d = QSIVolterra2D(filters=1, kernel_size=4, wave='haar')
qlayer3d = QSIVolterra3D(filters=1, kernel_size=4, wave='haar')
```

## Citation

This software is released for broad research, educational, and engineering use. If this package proves useful in related work, please cite the following thesis, whose Chapter 2 presents the underlying theory and computational details:

```bibtex
@misc{tarafdar2026interpretablefrugallearningsystems,
      title={Interpretable and Frugal Learning Systems Employing Multiresolution Pyramids and Volterra Kernels},
      author={Kishore Kumar Tarafdar},
      year={2026},
      eprint={2606.15011},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.15011},
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
