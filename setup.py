from setuptools import setup

setup(
    name="simbiofilm",
    version="0.1",
    description="Python biofilm simulation framework",
    author="Matt Simmons",
    author_email="matthew.simmons@dartmouth.edu",
    license="MIT",
    packages=["simbiofilm"],
    install_requires=["numpy", "fipy", "scipy", "scikit-fmm==0.0.9", "numba"],
)
