import numpy as np
from astropy.io import ascii
import h5py

"""
    Constants
    ----------
    file_path : str
        Path to the HDF5 file.
    ME : float
        Electron mass.
    CL : float
        Speed of light.
    HPL : float
        Planck constant.
    LSUN : float, optional
        Solar luminosity. Default is 3.827e33.
"""
LSUN = 3.827e33
CL = 2.99792458e10
ME = 9.1093826e-28
HPL =  6.6260693e-27

def load_spectrum(file):
    """
The cell below defines a helper function, `grmonty(file)`, which:
1. Reads a GRMONTY SED file with no header.
2. Extracts frequency bins and converts them to physical frequencies.
3. Builds arrays of luminosities for six viewing angles, converting from solar luminosities to cgs units.
5. Returns the frequency array and luminosity arrays for plotting

The notebook also defines the solid-angle weights (`domega_array`) associated with each of the six angular bins, which are typically used when computing angle-averaged spectra.

This setup provides a minimal, reproducible example for inspecting GRMONTY outputs and serves as a starting point for plotting spectra, comparing runs, or integrating over angles.

    Reads SEDs in the format provided by grmonty.
    
    Returns:
        nu: Frequency array
        ll: Luminosity array (log10)
        tauabs: Absorption optical depth
        domega: Array of solid angles for each theta bin
    """
    domega_arr = None
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("# dOmega:"):
                # Split by colon to get the numbers, then split by space
                parts = line.split(':')[1].strip().split()
                domega_arr = np.array([np.float64(x) for x in parts])
                break
    
    if domega_arr is None:
        print(f"Warning: No dOmega header found in {file}")

    s = ascii.read(file, format='no_header')

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

    nu = np.array(10**s['col1'] * (ME * CL**2/HPL))

    return nu, ll, tauabs, domega_arr


def load_igrmonty(file_path):
    """
    Load IGRMonty spectrum data from an HDF5 file and return internal variables.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    """
    with h5py.File(file_path, "r") as f:
        output_group = f["output"]

        nu_igr = 10 ** output_group["lnu"][:] * (ME * CL**2 / HPL)
        nuLnu_igr = output_group["nuLnu"][:] * LSUN
        dOmega_igr = output_group["dOmega"][:]

    return nu_igr, nuLnu_igr, dOmega_igr,