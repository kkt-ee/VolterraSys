from functools import lru_cache

import tensorflow as tf

from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_op import analysis_filterbank_axis, synthesis_filterbank_axis


_VALID_BACKENDS = ('matrix', 'filterbank')


def validate_backend(backend):
    if backend not in _VALID_BACKENDS:
        choices = ', '.join(repr(value) for value in _VALID_BACKENDS)
        raise ValueError(f"backend must be one of: {choices}.")
    return backend


def validate_level(level):
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer.")
    return level


@lru_cache(maxsize=None)
def _filters(wave):
    filters = FetchAnalysisSynthesisFilters(wave)
    analysis = tuple(tuple(values) for values in filters.analysis())
    synthesis = tuple(tuple(values) for values in filters.synthesis())
    return analysis, synthesis


def _validate_multilevel_shape(x, filters, axis, level):
    level = validate_level(level)
    rank = x.shape.rank
    if rank is None:
        raise ValueError("The input rank must be statically known.")
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"axis={axis} is invalid for rank-{rank} input.")

    length = x.shape[axis]
    if length is None:
        raise ValueError("The transformed length must be statically known.")
    length = int(length)
    divisor = 2 ** level
    if length % divisor:
        raise ValueError(
            f"The transformed length must be divisible by 2**level={divisor}."
        )
    if length // (2 ** (level - 1)) < len(filters[0]):
        raise ValueError(
            "Every decomposition level must cover the wavelet-filter length."
        )
    return axis, length


def _analysis_multilevel_axis(x, filters, axis, level):
    axis, _ = _validate_multilevel_shape(x, filters, axis, level)
    if level == 1:
        return analysis_filterbank_axis(x, *filters, axis=axis)

    highpasses = []
    current = x
    for _ in range(level):
        packed = analysis_filterbank_axis(
            current,
            *filters,
            axis=axis,
        )
        current, highpass = tf.split(packed, 2, axis=axis)
        highpasses.append(highpass)
    return tf.concat([current] + list(reversed(highpasses)), axis=axis)


def _synthesis_multilevel_axis(x, filters, axis, level):
    axis, length = _validate_multilevel_shape(x, filters, axis, level)
    if level == 1:
        return synthesis_filterbank_axis(x, *filters, axis=axis)

    lowest_length = length // (2 ** level)
    sizes = [lowest_length, lowest_length]
    sizes.extend(
        lowest_length * (2 ** exponent)
        for exponent in range(1, level)
    )
    lowpass, *highpasses = tf.split(x, sizes, axis=axis)
    current = lowpass
    for highpass in highpasses:
        current = synthesis_filterbank_axis(
            tf.concat([current, highpass], axis=axis),
            *filters,
            axis=axis,
        )
    return current


def analysis_axis(x, wave, axis, level=1):
    """Apply the packed multilevel analysis operator A_J along one axis."""
    analysis, _ = _filters(wave)
    return _analysis_multilevel_axis(x, analysis, axis, level)


def synthesis_axis(x, wave, axis, level=1):
    """Apply the packed multilevel synthesis operator S_J along one axis."""
    _, synthesis = _filters(wave)
    return _synthesis_multilevel_axis(x, synthesis, axis, level)


def synthesis_transpose_axis(x, wave, axis, level=1):
    """Apply S_J^T along a natural-domain kernel input axis."""
    _, synthesis = _filters(wave)
    return _analysis_multilevel_axis(x, synthesis, axis, level)


def analysis_transpose_axis(x, wave, axis, level=1):
    """Apply A_J^T along a transform-domain kernel input axis."""
    analysis, _ = _filters(wave)
    return _synthesis_multilevel_axis(x, analysis, axis, level)


def natural_to_wavelet_kernel(kernel, wave, order, level=1):
    """Convert h to H = A_output,J h S_input,J^(tensor order)."""
    level = validate_level(level)
    transformed = kernel
    for axis in range(1, order + 1):
        transformed = synthesis_transpose_axis(
            transformed,
            wave,
            axis=axis,
            level=level,
        )
    return analysis_axis(transformed, wave, axis=0, level=level)


def wavelet_to_natural_kernel(kernel, wave, order, level=1):
    """Convert H to h = S_output,J H A_input,J^(tensor order)."""
    level = validate_level(level)
    transformed = synthesis_axis(kernel, wave, axis=0, level=level)
    for axis in range(1, order + 1):
        transformed = analysis_transpose_axis(
            transformed,
            wave,
            axis=axis,
            level=level,
        )
    return transformed
