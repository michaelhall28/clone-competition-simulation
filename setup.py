from distutils.core import setup
import numpy
from Cython.Distutils import Extension
from Cython.Distutils import build_ext
import cython_gsl

setup(
    include_dirs=[numpy.get_include(), 'clone_competition_simulation'],
    name='clone-competition-simulation',
    # ext_modules = cythonize("clone_competition_simulation/diff_cell_functions.pyx"),
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("diff_cell_functions",
                             ["clone_competition_simulation/diff_cell_functions.pyx"],
                             libraries=cython_gsl.get_libraries(),
                             library_dirs=[cython_gsl.get_library_dir()],
                             include_dirs=[cython_gsl.get_cython_include_dir()])]
)
