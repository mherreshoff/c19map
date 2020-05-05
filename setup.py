from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

# Note: this fixes some seemingly Mac specific problems where the include directive gets ignored.
cflags = os.environ.get("CFLAGS", "")
os.environ["CFLAGS"] = cflags + " -I" + numpy.get_include()

setup(
    name='model_derivative',
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(
        "model_derivative.pyx",
        include_path=[numpy.get_include()],
        compiler_directives={
            'language_level': 3
        })
)
