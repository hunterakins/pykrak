from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()


setup(
    name="pykrak",
    version="1.0.0",
    description="Normal mode wave equation solver for stratified fluids",
    url="https://github.com/hunterakins/pykrak",
    author="F. Hunter Akins",
    author_email="fakins@ucsd.edu",
    classifiers=[
        "License :: GPL v3.0"
    ],
    packages=["pykrak"],
    python_requires=">=3.7, <4",
    install_requires=["numba", "numpy", "scipy", "matplotlib", "stdlib"],
)
    
    
