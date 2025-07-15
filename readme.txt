Python implementation of normal mode solver for underwater acoustic propagation.
This is more or less a translation of Michael Porter's KRAKEN (not KRAKENC).
I also provide an internal wave mode equation solver using the same numerical methods, as Richard Evans FORTRAN model WAVE.
The model implementation uses numba to compile the routines so that it has similar speeds as KRAKEN.

Some small differences:
- I continue to do bisection and Brent for all meshes instead of switching to secant method with deflation

A big difference:
- Elastic layers are not supported 

Code is under the pykrak folder, which has its own readme.
There are some examples in pykrak/examples as well. 

To install, you can use pip:
pip install pykrak

12/05/2024 Update:
I recently implemented a more direct translation of KRAKEN in order to include elastic layers at the surface (such as ice) and elastic layers in the bottom. 
This differs from my previous code, which did not shoot from the bottom to the surface as in KRAKEN. 
These are in krak_routines.py, exposed through the list_input_solve function. This function is nearly identical to KRAKEN. It is benchmarked on some example environments in the tests subfolder.
