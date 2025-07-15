from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()


setup(
    name="pykrak",
    version="2.0.3",
    description="Normal mode wave equation solver for stratified fluids",
    url="https://github.com/hunterakins/pykrak",
    author="F. Hunter Akins",
    author_email="fakins@ucsd.edu",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    packages=["pykrak"],
    python_requires=">=3.9, <4",
    install_requires=["numba", "numpy", "scipy", "matplotlib"],
)
    
    
