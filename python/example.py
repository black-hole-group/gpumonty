import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

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
nu, nuLnu, tauabs, domega_array = grmonty('./output/example.spec')
# print("Read dOmega:", domega_array)



nuLnu_all = (nuLnu * domega_array[:, None] / (4 * np.pi)).sum(axis=0)

plt.figure(figsize=(7.5, 5.5))

plt.loglog(
    nu,
    nuLnu_all,
    linewidth=2.5,
    label=r'Angle-averaged spectrum',
    color='darkred'
)

plt.xlim(1e8, 1e20)

plt.xlabel(r'Frequency $\nu$ [Hz]', fontsize=16)
plt.ylabel(r'$\nu L_\nu$ [erg s$^{-1}$]', fontsize=16)

plt.legend(frameon=False, fontsize=16)

plt.grid(which='major', linestyle='-', alpha=0.7)
plt.grid(which='minor', linestyle=':', alpha=0.7)

plt.tick_params(
    axis='both',
    which='major',
    direction='in',
    length=8,
    width=1.5,
    labelsize=14,
    top=True,
    right=True
)

plt.tick_params(
    axis='both',
    which='minor',
    direction='in',
    length=4,
    width=1.2,
    top=True,
    right=True
)

plt.tight_layout()
plt.savefig(
    "./output/test_figure.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()