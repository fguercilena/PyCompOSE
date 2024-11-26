from setuptools import setup, Extension

# run with python setup.py build

# Compile *NQTLib.cpp* into a shared library 
setup(
    #...
    ext_modules=[Extension('NQTLib', ['NQTLib.cpp'],),],
)
