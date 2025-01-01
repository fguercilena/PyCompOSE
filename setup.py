from setuptools import setup, Extension

setup(
    name="PyCompOSE",
    version="0.9",
    description="Read and manipulate equation of state tables in CompOSE ASCII format",
    url="https://github.com/computationalrelativity/PyCompOSE",
    package_dir={"compose": "compose", "compose.NQTLib": "compose/NQTs"},
    packages=["compose", "compose.NQTLib"],
    ext_modules=[
        Extension(
            "compose.NQTLib",
            ["compose/NQTs/NQTLib.cpp"],
        ),
    ],
    install_requires=["h5py", "numpy", "scipy"],
)
