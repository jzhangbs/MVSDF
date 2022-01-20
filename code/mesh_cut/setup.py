from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("mesh_cut_ext", [
        "mesh_cut_ext.cpp",
        "IBFS/ibfs.cpp",
    ]),
]

setup(
    name="mesh_cut_ext",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
