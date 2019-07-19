
import glob
import pybind11
import petsc4py
import mpi4py
import os
import subprocess
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import distutils.ccompiler

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

VERSION = "2019.2.0.dev0"
RESTRICT_REQUIREMENTS = ">=2019.2.0.dev0,<2019.3"

REQUIREMENTS = [
    "numpy",
    "mpi4py",
    "petsc4py",
    "fenics-ffc",
    "fenics-ufl{}".format(RESTRICT_REQUIREMENTS),
]


def _pkgconfig_query(s):
    pkg_config_exe = os.environ.get('PKG_CONFIG', None) or 'pkg-config'
    cmd = [pkg_config_exe] + s.split()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    rc = proc.returncode
    return (rc, out.rstrip().decode('utf-8'))


def exists(package):
    "Test for the existence of a pkg-config file for a named package"
    return (_pkgconfig_query("--exists " + package)[0] == 0)


def parse(package):
    "Return a dict containing compile-time definitions"
    parse_map = {
        '-D': 'define_macros',
        '-I': 'include_dirs',
        '-L': 'library_dirs',
        '-l': 'libraries'
    }

    result = {x: [] for x in parse_map.values()}

    # Execute the query to pkg-config and clean the result.
    out = _pkgconfig_query(package + ' --cflags --libs')[1]
    out = out.replace('\\"', '')

    # Iterate through each token in the output.
    for token in out.split():
        key = parse_map.get(token[:2])
        if key:
            t = token[2:].strip()
            result[key].append(t)

    return result


if (exists('dolfin')):
    dolfin_pkg = parse('dolfin')
else:
    raise Exception("Can't find libdolfin pkgconfig")

includes = dolfin_pkg['include_dirs']
includes += [pybind11.get_include()]
includes += [mpi4py.get_include()]
includes += [petsc4py.get_include()]

libdirs = dolfin_pkg['library_dirs']
libraries = dolfin_pkg['libraries']

defines = dolfin_pkg['define_macros']
defines = [tuple(d.split("=")) for d in defines]
for i, d in enumerate(defines):
    if len(d) == 1:
        defines[i] = (d[0], '1')
    elif "." in d[1]:
        defines[i] = (d[0], '\"' + d[1] + '\"')

cpp_files = glob.glob("src/*.cpp")

ext_modules = [Extension('dolfin.cpp',
                         cpp_files,
                         define_macros=defines,
                         include_dirs=includes,
                         library_dirs=libdirs,
                         libraries=libraries,
                         language='c++')]


def parallelCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                    extra_preargs=None, extra_postargs=None, depends=None):
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs,
                                                                          sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    import multiprocessing
    import multiprocessing.pool

    if "CI" in os.environ:
        N = 2
    else:
        # Use half of processor count (probably best, in case of hyperthreading)
        N = multiprocessing.cpu_count() // 2

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile, objects))
    return objects


distutils.ccompiler.CCompiler.compile = parallelCompile


class BuildExt(build_ext):

    def build_extensions(self, *args, **kwargs):
        opts = ['-std=c++14', '-g0']
        link_opts = ['-Wl,-rpath,' + ldir for ldir in libdirs]
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(name='fenics-dolfin',
      version=VERSION,
      author='FEniCS Project',
      description='DOLFIN Python interface',
      long_description='',
      packages=["dolfin",
                "dolfin.function",
                "dolfin.fem",
                "dolfin.la",
                "dolfin_utils.test"],
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExt},
      install_requires=REQUIREMENTS,
      zip_safe=False)
