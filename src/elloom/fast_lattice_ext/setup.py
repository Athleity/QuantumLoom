from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Compiler flags for optimization
extra_compile_args = []
if sys.platform == 'win32':
    extra_compile_args = ['/O2', '/std:c++20']
else:
    extra_compile_args = ['-O3', '-std=c++20', '-march=native']

ext_modules = [
    Pybind11Extension(
        "fast_lattice",
        ["fast_lattice.cpp"],
        extra_compile_args=extra_compile_args,
        cxx_std=20,
    ),
]

setup(
    name="fast_lattice",
    version="0.1.0",
    author="Your Name",
    description="High-performance lattice indexing for QEC",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.10.0",
        "numpy>=1.20.0",
    ],
)