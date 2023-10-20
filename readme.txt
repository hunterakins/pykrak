Python implementation of normal mode solver for underwater acoustic propagation.
This is more or less a translation of Michael Porter's KRAKEN (not KRAKENC).
I also provide an internal wave mode equation solver using the same numerical methods, as Richard Evans FORTRAN model WAVE.
The model implementation uses numba to compile the routines so that it has similar speeds as KRAKEN.

Some small differences:
- I continue to do bisection and Brent for all meshes instead of switching to secant method with deflation
- Coupled mode here has continuity of particle velocity as well as pressure between segments

A big difference:
- Elastic layers are not supported 

Code is under the pykrak folder, which has its own readme.
There are some examples in pykrak/examples as well. 

To install, you can use pip:
pip install pykrak



