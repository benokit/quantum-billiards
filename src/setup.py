from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "cython_functions",
        [r"BasisModules/cython_functions.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],

    )
]

setup(
    name='cython_functions',
    ext_modules = cythonize(ext_modules),
    language="c++",
)
