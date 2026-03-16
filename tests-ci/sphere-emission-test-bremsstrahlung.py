import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

#check if compilation is sphere
#check if nu emission sphere is between 1e8 1e24
#check if Ne = 1e13, Thetae = 100, B = 1.
#put no scattering, so scattering from photons = 1 and max_layer_sca = 1.

#read from other models
def grmonty(file):
    """
    Reads SEDs in the format provided by grmonty.
    
    Returns:
        nu: Frequency array
        ll: Luminosity array (log10)
        tauabs: Absorption optical depth
        domega: Array of solid angles for each theta bin
    """
    LSUN = 3.827e33
    CL = 2.99792458e10
    ME = 9.1093826e-28
    HPL =  6.6260693e-27

    # 1. Parse the header for dOmega
    domega_arr = None
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("# dOmega:"):
                # Split by colon to get the numbers, then split by space
                parts = line.split(':')[1].strip().split()
                domega_arr = np.array([np.float64(x) for x in parts])
                break
    
    # Check if dOmega was found; if not, you might want a default or error
    if domega_arr is None:
        print(f"Warning: No dOmega header found in {file}")

    # 2. Read the numerical data
    # format='no_header' treats lines starting with # as comments, 
    # so our new header won't break this.
    s = ascii.read(file, format='no_header')

    # Array conversions for luminosity values (log10)
    # Note: This hardcoding of 6 bins assumes N_THBINS is always 6.
    # If N_THBINS changes, this section needs to be dynamic based on domega_arr.size
    ll = np.zeros((6, len(s))) 
    ll[0] = np.array(s['col2'] * LSUN)
    ll[1] = np.array(s['col8'] * LSUN)
    ll[2] = np.array(s['col14']* LSUN)
    ll[3] = np.array(s['col20']* LSUN)
    ll[4] = np.array(s['col26']* LSUN)
    ll[5] = np.array(s['col32']* LSUN)
    
    tauabs = np.zeros((6, len(s)))
    tauabs[0] = np.array(s['col3'])     
    tauabs[1] = np.array(s['col9'])
    tauabs[2] = np.array(s['col15'])
    tauabs[3] = np.array(s['col21'])
    tauabs[4] = np.array(s['col27'])
    tauabs[5] = np.array(s['col33'])

    # Compute frequency (nu) values
    # The C code outputs log10(energy), so we convert back
    nu = np.array(10**s['col1'] * (ME * CL**2/HPL))

    return nu, ll, tauabs, domega_arr

# Example usage:
nu, nuLnu, tauabs, domega_array = grmonty('./output/test_sphere_emissivity_bremsstrahlung.spec')
print("Read dOmega:", domega_array)


import numpy as np
from scipy.special import kn  
from scipy.integrate import quad

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