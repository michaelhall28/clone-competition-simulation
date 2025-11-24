# from setuptools import setup, Extension
# import numpy
# from Cython.Distutils import Extension
# from Cython.Distutils import build_ext
# import cython_gsl
#
# # Hardcoded path for Homebrew GSL (Apple Silicon)
# # This is where the .h files live
# gsl_include_dir = "/opt/homebrew/include"
# # This is where the .dylib files live
# gsl_lib_dir = "/opt/homebrew/lib"
#
# # setup(
# #     include_dirs=[numpy.get_include(), 'clone_competition_simulation'],
# #     name='clone-competition-simulation',
# #     # ext_modules = cythonize("clone_competition_simulation/diff_cell_functions.pyx"),
# #     cmdclass = {'build_ext': build_ext},
# #     ext_modules = [Extension("diff_cell_functions",
# #                              ["clone_competition_simulation/diff_cell_functions.pyx"],
# #                              libraries=['gsl', 'gslcblas'], #cython_gsl.get_libraries(),
# #                              library_dirs=["/opt/homebrew/Cellar/gsl/2.8/lib"], #[cython_gsl.get_library_dir()],
# #                              include_dirs=["/Users/michaelhall/dev/clone-competition-simulation/.venv/lib/python3.10/site-packages", gsl_include_dir]
# #                              #[cython_gsl.get_cython_include_dir()]
# #                              )]
# # )


from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import subprocess

# Function to ask gsl-config where files are
def get_gsl_config(option):
    try:
        result = subprocess.check_output(["gsl-config", option], text=True).strip()
        # gsl-config returns "-L/path/to/lib", we just want "/path/to/lib"
        if option == "--libs":
            # specific parsing might be needed depending on output
            return result.split("-L")[1].split(" ")[0]
        return result.replace("-I", "") # remove -I flag for include
    except (FileNotFoundError, IndexError):
        return None # Fallback or error out

gsl_include_dir = get_gsl_config("--cflags") or "/usr/include"
gsl_lib_dir = get_gsl_config("--libs") or "/usr/lib"

extensions = [
    Extension(
        name="diff_cell_functions",
        sources=["clone_competition_simulation/diff_cell_functions.pyx"],

        # 1. TELL COMPILER WHERE HEADERS ARE
        # This fixes "gsl/gsl_mode.h not found"
        include_dirs=[
            numpy.get_include(),
            gsl_include_dir,
            "clone_competition_simulation"
        ],

        # 2. TELL LINKER WHERE LIBRARIES ARE
        # Only paths here, no "-l" flags
        library_dirs=[gsl_lib_dir],

        # 3. TELL LINKER WHICH LIBRARIES TO LINK
        libraries=["gsl", "gslcblas"],
    )
]

setup(
    name='clone-competition-simulation',
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
)