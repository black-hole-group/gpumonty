import numpy as np
import sys

np.set_printoptions(threshold=np.nan) # descomentar para printar tudo na tela

"""
You must write the three dimensions (r, theta, phi) - IN THIS ORDER - followed by ONLY THE NUMBERS of the desired dump file
Example: $ python bin2ascii.py 256 128 64 023
This means your grid is 256x128x64 and the file you are working on is dump023
"""

N1, N2, N3, filenumber = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

bindata   = "dump"+filenumber
asciidata = "ascii"+filenumber

def read_body(dump, ascii, nx = None, ny = None, nz = None, noround = False):

  binfile   = open(dump, "rb")
   asciifile = open(ascii, "w")

  header = binfile.readline()
   asciifile.write(header)

  body = np.fromfile(binfile, dtype = np.float32, count = -1) # sasha
   gd   = body.view().reshape((-1, nz, ny, nx), order = 'F')   # sasha
   gd   = np.float32(gd.transpose(0, 3, 2, 1))                 # sasha

  print gd.shape

"""
    dV = nx*ny*nz
    cols = len(body)/dV
    mat = body.view().reshape((cols, dV), order = 'F')
    mat = np.float32(mat.transpose())
    print len(mat)
    np.savetxt(asciifile, mat, fmt = '%.18e')
"""

  return gd

asciidata = read_body(bindata, asciidata, int(N1), int(N2), int(N3), noround = 1)
    

"""
    for i in N1:
        for j in N2:
            for j in N3:
                while (line % 42 !=0):
                    ascii.write(gd(i,j,k))

   for dataslice in gd:
        np.savetxt(asciifile, dataslice, fmt = '%.18e')

   asciifile.write(gd)

"""
