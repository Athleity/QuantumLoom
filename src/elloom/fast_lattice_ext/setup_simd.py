from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile_args = []
if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/std:c++20', '/arch:AVX2', '/openmp']
else:
    extra_compile_args = ['-O3', '-std=c++20', '-mavx2', '-fopenmp']

ext_modules = [
    Pybind11Extension(
        "fast_lattice_simd",
        ["fast_lattice_simd.cpp"],
        extra_compile_args=extra_compile_args,
        cxx_std=20,
    ),
]

setup(
    name="fast_lattice_simd",
    version="0.2.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)