from setuptools import setup, Extension

setup(
    name="PyCompOSE",
    setup_requires=["setuptools_scm"],
    use_scm_version={"write_to": "compose/_version.py"},
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
