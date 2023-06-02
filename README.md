Python implementation of normal mode solver for underwater acoustic propagation.
More or less a translation of Michael Porter's KRAKEN.
Some small differences:
- I continue to do bisection and Brent for all meshes instead of switching to secant method with deflation
- I do not have partial pivoting in my inverse iteration tridiagonal solver (at this point)

A big difference:
- Elastic layers are not supported 

File list:
-pynm_env.py
    The Env object used to manage all of the environmental parameters
    and to run the forward model, save results
-sturm_seq.py
    The numerical code for solving the sturm seq eigenvalue problem
-mesh_routines.py
    Code for creating meshes
- shooting_routines.py
    Shooting through layers
- inverse_iterations.py
    Solving for eigenvectors from eigenvalues
-pressure_calc.py
    Some helper functions for computing pressure fields using the modal sum
-group_pert.py
    Calculating modal group speed from perturbation theory
-attn_pert.py
    Calculating attenuation values using perturbation theory
-coupled_modes.py
    Coupled mode field calc (in work)
-env_pert.py
    PertEnv object, inherits Env
    Adds some extra methods for managing making changes to the
    Env using parameterizations
    Also computes first order corrections to wavenumbers using 
    perturbation theory
misc.py
    Some numerical integrations
-pekeris_test.py 
    compare with KRAKEN (not tracked by git)
-pekeris_comp.py 
    compare with KRAKEN (not tracked by git)


