# setup.py
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='bitnet_cpp',
      ext_modules=[cpp_extension.CppExtension('bitnet_cpp', ['bitnet_ops.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})