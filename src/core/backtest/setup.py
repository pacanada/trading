from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

ext_modules = [
    Extension(
        name="backtesting_cpp",
        sources=["backtesting_cpp.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O3", "-march=native", '-stdlib=libc++', '-std=c++11'],
        language="c++",
        include_dirs=[".", numpy.get_include()],
    )
]

setup(
    name="sum", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)