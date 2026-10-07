from functools import lru_cache

from TFDWT.DWTFilters import FetchAnalysisSynthesisFilters
from TFDWT.dwt_op import analysis_filterbank_axis, synthesis_filterbank_axis


_VALID_BACKENDS = ('matrix', 'filterbank')


def validate_backend(backend):
    if backend not in _VALID_BACKENDS:
        choices = ', '.join(repr(value) for value in _VALID_BACKENDS)
        raise ValueError(f"backend must be one of: {choices}.")
    return backend


@lru_cache(maxsize=None)
def _filters(wave):
    filters = FetchAnalysisSynthesisFilters(wave)
    analysis = tuple(tuple(values) for values in filters.analysis())
    synthesis = tuple(tuple(values) for values in filters.synthesis())
    return analysis, synthesis


def analysis_axis(x, wave, axis):
    """Apply the wavelet analysis operator A along one tensor axis."""
    analysis, _ = _filters(wave)
    return analysis_filterbank_axis(x, *analysis, axis=axis)


def synthesis_axis(x, wave, axis):
    """Apply the wavelet synthesis operator S along one tensor axis."""
    _, synthesis = _filters(wave)
    return synthesis_filterbank_axis(x, *synthesis, axis=axis)


def synthesis_transpose_axis(x, wave, axis):
    """Apply S^T along a natural-domain kernel input axis."""
    _, synthesis = _filters(wave)
    return analysis_filterbank_axis(x, *synthesis, axis=axis)


def analysis_transpose_axis(x, wave, axis):
    """Apply A^T along a transform-domain kernel input axis."""
    analysis, _ = _filters(wave)
    return synthesis_filterbank_axis(x, *analysis, axis=axis)


def natural_to_wavelet_kernel(kernel, wave, order):
    """Convert h to H = A_output h S_input^(tensor order)."""
    transformed = kernel
    for axis in range(1, order + 1):
        transformed = synthesis_transpose_axis(
            transformed,
            wave,
            axis=axis,
        )
    return analysis_axis(transformed, wave, axis=0)


def wavelet_to_natural_kernel(kernel, wave, order):
    """Convert H to h = S_output H A_input^(tensor order)."""
    transformed = synthesis_axis(kernel, wave, axis=0)
    for axis in range(1, order + 1):
        transformed = analysis_transpose_axis(
            transformed,
            wave,
            axis=axis,
        )
    return transformed
