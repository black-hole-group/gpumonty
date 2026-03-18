import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

from python.example import grmonty

#check if compilation is sphere
#check if nu emission sphere is between 1e8 1e24
#check if Ne = 1e13, Thetae = 100, B = 1.
#put no scattering, so scattering from photons = 1 and max_layer_sca = 1.

#read from other models
LSUN = 3.827e33
CL = 2.99792458e10
ME = 9.1093826e-28
HPL =  6.6260693e-27
# Open the HDF5 file
import h5py
with h5py.File('./output/test_sphere_emissivity_bremsstrahlung.h5', 'r') as f:
#with h5py.File('../../igrmonty/spectrum.h5', 'r') as f:
    # Access the 'output' group
    output_group = f['output']
    # Extract the datasets 'lnu' and 'nulnu'
    nu = 10**output_group['lnu'][:] * (ME * CL**2/HPL)
    nuLnu = output_group['nuLnu'][:] * LSUN
    domega_array = output_group['dOmega'][:]
    

# Constants

ME = 9.1093826e-28
CL = 2.99792458e10
HPL = 6.6260693e-27



def jnu_bremss(Ne, Thetae, nu):
    if Thetae < 0.3:
        return np.zeros_like(nu)

    KBOL = 1.3806505e-16
    BREMS_FAC = 6.533236526124812e-39

    Te = Thetae * ME * CL * CL / KBOL
    x = HPL * nu / (KBOL * Te)

    efac = np.where(
        x < 1e-3,
        (24.0 - 24.0*x + 12.0*x**2 - 4.0*x**3 + x**4) / 24.0,
        np.exp(-x)
    )

    rel = 1.0 + 4.4e-10 * Te

    jv = BREMS_FAC * (1.0 / np.sqrt(Te)) * Ne**2 * efac * rel

    return jv

    
def int_jnu_bremss(Ne, thetae, nu):
    return 4 * np.pi * jnu_bremss(Ne, thetae, nu)

R = 1.0000
Rin = 0.01

#jnu_values = int_jnu(1e13, 100, 1, np.float128(nu))
jnu_values = int_jnu_bremss(1e13, 100, np.float128(nu))

dv = 4/3 * np.pi * (R**3 - Rin**3)

Luminosity_analytic = jnu_values
exp_approx = np.zeros_like(nu)

xdata = nu
y_simdata = (nuLnu * domega_array[:, None] / (4 * np.pi)).sum(0)
y_analyticdata = Luminosity_analytic * nu * dv

# --- find closest indices to frequency limits ---
nu_min = 1e8
nu_max = 1e23

idx_min = np.argmin(np.abs(nu - nu_min))
idx_max = np.argmin(np.abs(nu - nu_max))

# ensure correct ordering
if idx_min > idx_max:
    idx_min, idx_max = idx_max, idx_min

# --- slice arrays ---
xdata = xdata[idx_min:idx_max+1]
y_simdata = y_simdata[idx_min:idx_max+1]
y_analyticdata = y_analyticdata[idx_min:idx_max+1]

relative_diff = (y_simdata - y_analyticdata) / y_analyticdata

tol = 0.05

rel_err = np.abs(relative_diff)
max_err = rel_err.max()

print("Maximum relative error:", max_err)

if not np.all(rel_err < tol):
    bad_idx = np.where(rel_err >= tol)[0]
    print("\033[91mIndices failing tolerance:\033[0m", bad_idx[:10])
    print("\033[91mErrors:\033[0m", rel_err[bad_idx[:10]])
    raise RuntimeError(f"Test failed: relative error > {tol}")
else:
    print("\033[92mTest passed: all errors within 5%\033[0m")