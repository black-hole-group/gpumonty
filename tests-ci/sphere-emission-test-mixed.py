import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

#check if compilation is sphere
#check if nu emission sphere is between 1e8 1e16
#check if Ne = 1e13, Thetae = 100, B = 1.
#put no scattering, so scattering from photons = 1 and max_layer_sca = 1.

import h5py
with h5py.File('./output/test_sphere_emissivity_mixed.h5', 'r') as f:
#with h5py.File('../../igrmonty/spectrum.h5', 'r') as f:
    # Access the 'output' group
    output_group = f['output']
    # Extract the datasets 'lnu' and 'nulnu'
    nu = 10**output_group['lnu'][:] * (ME * CL**2/HPL)
    nuLnu = output_group['nuLnu'][:] * LSUN
    domega_array = output_group['dOmega'][:]


import numpy as np
from scipy.special import kn  
from scipy.integrate import quad

# Constants
M_SQRT2 = np.sqrt(2)
EE = 4.80320680e-10
ME = 9.1093826e-28
CL = 2.99792458e10
JCST = M_SQRT2 * EE**3 / (27 * ME * CL**2)
CST = 1.88774862536  # 2^(11/12)
CL = 2.99792458e10  
ME = 9.1093826e-28
HPL = 6.6260693e-27
THETAE_MIN = 0.3

KMIN = 0.002
KMAX = 1e7
TMIN = 0.3
TMAX = 1e2
N_ESAMP = 2500
EPSABS = 0.
EPSREL = 1e-6
dlT = 1/(np.log(TMAX/TMIN)/N_ESAMP)
lT_min = np.log(TMIN)
F_table = np.zeros(N_ESAMP + 1)
K2_table = np.zeros(N_ESAMP + 1)

def jnu_integrand(th, K):
    """Integrand for the emissivity calculation."""
    sth = np.sin(th)
    x = K / sth
    if sth < 1.e-150 or x > 2.e8:
        return 0.
    return sth**2 * (np.sqrt(x) + CST * x**(1. / 6.))**2 * np.exp(-x**(1. / 3.))

def init_emiss_tables():
    """Initialize emissivity and Bessel function tables."""
    global F_table, K2_table
    
    dlK = np.log(KMAX / KMIN) / N_ESAMP
    dlT = np.log(TMAX / TMIN) / N_ESAMP

    for k in range(N_ESAMP + 1):
        K = np.exp(k * dlK + np.log(KMIN))
        result, err = quad(jnu_integrand, 0, np.pi / 2, args=(K), epsabs=EPSABS, epsrel=EPSREL)
        F_table[k] = np.log(4 * np.pi * result)

    for k in range(N_ESAMP + 1):
        T = np.exp(k * dlT + np.log(TMIN))
        K2_table[k] = np.log(kn(2, 1. / T))

    global inv_dlK, inv_dlT
    inv_dlK = 1. / dlK
    inv_dlT = 1. / dlT


def K2_eval(Thetae):
    """Evaluate K2 using precomputed values and interpolation."""
    if Thetae < TMIN:
        return 0.
    elif Thetae > TMAX:
        return 2. * Thetae * Thetae  # As per CUDA code
    
    if(Thetae == 100):
        return 19999.50006838941
    return linear_interp_K2(Thetae)

def linear_interp_K2(Thetae):
    """Linear interpolation for K2 evaluation."""
    lT = np.log(Thetae)
    
    di = (lT - lT_min) * inv_dlT 
    i = int(di)
    di = di - i  
    print(i, di)
    return np.exp((1. - di) * K2_table[i] + di * K2_table[i + 1])
    
KFAC = 9 * np.pi * ME * CL / EE

def linear_interp_F(K):
    """Linearly interpolate F_table for a given K value."""
    lK_min = np.log(KMIN)
    dlK = np.log(KMAX / KMIN) / N_ESAMP
    inv_dlK = 1.0 / dlK

    lK = np.log(K)
    di = (lK - lK_min) * inv_dlK
    i = int(di)
    di -= i

    if i < 0 or i >= N_ESAMP:
        return 0.

    result = np.exp((1.0 - di) * F_table[i] + di * F_table[i + 1])
    return result

def F_eval(Thetae, Bmag, nu):
    """Evaluate F using precomputed tables or approximation based on K."""
    K = KFAC * nu / (Bmag * Thetae**2)
    if K > KMAX:
        return 0.
    elif K < KMIN:
        x = K**(1 / 3)
        return x * (37.67503800178 + 2.240274341836 * x)
    else:
        return linear_interp_F(K)

def Bnu(nu, Thetae):
    """
    Calculates the Planck function B_nu at a given frequency nu and electron temperature Thetae.
    
    Parameters:
    nu (array-like): Frequency in Hz (can be an array).
    Thetae (float): Dimensionless electron temperature (Thetae = kT_e / m_e c^2).
    
    Returns:
    array: B_nu values for the given frequencies.
    """
    nu = np.array(nu, dtype=np.float64)
    
    x = HPL * nu / (ME * CL**2 * Thetae)

    result = np.zeros_like(x)
    
    mask_low = x < 1e-8
    result[mask_low] = (2. * HPL * nu[mask_low]**3 / CL**2) / (
        x[mask_low] / 24. * (24. + x[mask_low] * (12. + x[mask_low] * (4. + x[mask_low])))
    )
    
    mask_high = ~mask_low
    result[mask_high] = (2.0 * HPL * nu[mask_high]**3 / CL**2) / (np.float128(np.exp(np.float128(x[mask_high]))) - 1.0)
    
    return result

# Synchrotron emissivity function
def jnu(ne, nu, B, thetae):
    sin_value = 1
    nus = (2/9) * (EE * B/(2 * np.pi * ME * CL)) * thetae**2 * sin_value
    X = nu / nus
    factor = np.sqrt(2) * np.pi * EE**2 * ne * nus / (3 * CL * kn(2, 1/thetae))
    return factor * (X**(1/2) + 2**(11/12) * X**(1/6))**2 * np.exp(-X**(1/3))

def int_jnu(Ne, Thetae, Bmag, nu):
    """Calculate emissivity jnu using precomputed tables and F_eval, allowing nu to be an array."""
    nu = np.atleast_1d(nu)

    if Thetae < THETAE_MIN:
        return np.zeros_like(nu)

    K2 = K2_eval(Thetae)
    if K2 == 0.:
        return np.zeros_like(nu)
    
    j_fac = Ne * Bmag * Thetae**2 / K2

    F_vals = np.array([F_eval(Thetae, Bmag, n) for n in nu])
    jnu_vals = JCST * j_fac * F_vals

    return jnu_vals[0] if jnu_vals.size == 1 else jnu_vals


# Initialize tables before calculations
init_emiss_tables()



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

jnu_values = int_jnu(1e13, 100, 1, np.float128(nu)) + int_jnu_bremss(1e13, 100, np.float128(nu))

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