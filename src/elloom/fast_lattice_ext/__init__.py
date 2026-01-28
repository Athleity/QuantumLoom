"""Advanced lattice optimizations with SIMD and GPU support"""

try:
    from .fast_lattice_simd import FastLatticeSIMD
    __all__ = ['FastLatticeSIMD']
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import SIMD module: {e}")
    __all__ = []