from setuptools import setup, Extension
import pybind11

ext = Extension(
    'profile_rng',
    ['profile_rng_py.cpp', 'entropy_source.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=['/std:c++20', '/O2', '/arch:AVX2'],
)

setup(name='profile_rng', ext_modules=[ext], zip_safe=False)
