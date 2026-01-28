from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "entropy_core.entropy_core",
        [
            "src/entropy_core/fast_lattice_entropy.cpp",
            "src/entropy_core/entropy_source.cpp",
        ],
        include_dirs=["src/entropy_core"],
        cxx_std=17,
        extra_compile_args=["/O2", "/arch:AVX2", "/fp:fast"],
    ),
]

setup(
    name="entropy-core",
    version="1.0.0",
    packages=["entropy_core"],
    package_dir={"entropy_core": "src/entropy_core"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
