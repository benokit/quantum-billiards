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

ext_modules = [
    Extension(
        "husimi_cy",
        [r"CoreModules/husimi_cy.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],

    )
]

setup(
    name='husimi_cy',
    ext_modules = cythonize(ext_modules),
    language="c++",
)


ext_modules = [
    Extension(
        "spectral_functions",
        [r"AnalysisModules/spectral_functions.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],

    )
]

setup(
    name='spectral_functions',
    ext_modules = cythonize(ext_modules),
    language="c++",
)
