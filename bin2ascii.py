import numpy as np
import sys

"""
You must write the three dimensions (r, theta, phi) - IN THIS ORDER - followed by ONLY THE NUMBERS of the desired dump file

Examples: 
$ python bin2ascii.py 256 128 64 023

This means your grid is 256x128x64 and the file you are working on is dump023

$ python bin2ascii.py 288 128 64 1174

This means your grid is 288x128x64 and the file you are working on is fieldline1174
"""

N1, N2, N3, filenumber = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

#bindata   = "fieldline"+filenumber+".bin"
bindata   = "dump"+filenumber
asciidata = "ascii"+filenumber

def read_body(dump, ascii, nx = None, ny = None, nz = None, noround = False):

    binfile   = open(dump, "rb")
    asciifile = open(ascii, "w")

    header = binfile.readline()
    asciifile.write(header)

    body = np.fromfile(binfile, dtype = np.float32, count = -1)
    dV = nx*ny*nz
    cols = len(body)/dV
    mat = body.view().reshape((cols, dV), order = 'F')
    mat = np.float32(mat.transpose())
    np.savetxt(asciifile, mat, fmt = '%.18e')

    return mat

theasciidata = read_body(bindata, asciidata, int(N1), int(N2), int(N3), noround = 1)


# the following 2 commented lines are here for "preservation reasons" 
# they're taken from sasha's script but are not necessary here
# they appear right below body = np.fromfile...
#    gd   = body.view().reshape((-1, nz, ny, nx), order = 'F')
#    gd   = np.float32(gd.transpose(0, 3, 2, 1))
