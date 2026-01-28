import os
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'fast_lattice_entropy',
        ['fast_lattice_entropy.cpp', 'entropy_source.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/std:c++20', '/O2', '/arch:AVX2'] if os.name == 'nt' 
                           else ['-std=c++20', '-O2', '-mavx2'],
    ),
]

setup(
    name='fast_lattice_entropy',
    ext_modules=ext_modules,
    zip_safe=False,
)
