from setuptools import setup, Extension
import pybind11
import sys

# Determine compiler flags
if sys.platform == 'win32':
    extra_compile_args = [
        '/O2',           # Maximum optimization
        '/std:c++17',    # C++17 standard
        '/arch:AVX2',    # AVX2 instructions
        '/fp:fast',      # Fast floating point
    ]
    extra_link_args = []
else:
    extra_compile_args = [
        '-O3',           # Maximum optimization
        '-std=c++17',    # C++17 standard
        '-march=native', # Use all CPU features
        '-ffast-math',   # Fast math
    ]
    extra_link_args = []

ext_modules = [
    Extension(
        'fast_lattice_entropy',
        ['fast_lattice_entropy.cpp', 'entropy_source.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='fast_lattice_entropy',
    version='1.0.0',
    ext_modules=ext_modules,
    zip_safe=False,
)
