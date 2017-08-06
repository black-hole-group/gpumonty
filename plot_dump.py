# -*- coding: utf8 -*-

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nmmn.lsd
import math


listr   = []
listth  = []
listx1  = [] # ln(r)
listx2  = []
listrho = []
listu   = []
listvr  = []
listvth = []
listvph = []
listBr  = []
listBth = []
listBph = []
listx = []
listy = []

f = open('dump019', 'r')
header = f.readline()

for line in f:
    col = line.split()
    listx1.append(float(col[0]))
    listx2.append(float(col[1]))
    listr.append(float(col[2]))
    listth.append(float(col[3]))
    listrho.append(float(col[4]))
    listu.append(float(col[5]))
    listvr.append(float(col[6]))
    listvth.append(float(col[7]))
    listvph.append(float(col[8]))
    listBr.append(float(col[9]))
    listBth.append(float(col[10]))
    listBph.append(float(col[11]))
#    listx.append(float(col[2]) * np.cos(float(col[3])))
#    listy.append(float(col[2]) * np.sin(float(col[3])))
    listx.append(float(col[2]) * np.cos(float(col[3]) - np.pi/2.))
    listy.append(float(col[2]) * np.sin(float(col[3]) - np.pi/2.))

f.close()

r   = np.array(listr)
th  = np.array(listth)
rho = np.array(listrho)
vr  = np.array(listvr)
vth = np.array(listvth)
vph = np.array(listvph)
Br  = np.array(listBr)
Bth = np.array(listBth)
Bph = np.array(listBph)
x = np.array(listx)
y = np.array(listy)

#print('r_min  = %s') %np.amin(r)
#print('r_max  = %s') %np.amax(r)
#print('th_min = %s') %np.amin(th)
#print('th_max = %s') %np.amax(th)

def prepare_regrid(x, y):
    xsize = 256
    ysize = 256 
    fact = 5 # "aumento de resolução" para o espaçamento grid novo
    newdx = fact*xsize
    newdy = fact*ysize
    xnew = np.linspace(round(np.amin(x)), round(np.amax(x)), newdx)
    ynew = np.linspace(round(np.amin(y)), round(np.amax(y)), newdy)   
    return xnew, ynew

def newgrid(x, y, z, xnew, ynew):
    znew = scipy.interpolate.griddata((x, y), z, (xnew[None,:], ynew[:,None]), method='cubic')
    return znew

xnew, ynew = prepare_regrid(x, y)
newrho = newgrid(x, y, rho, xnew, ynew)

plt.imshow(np.log10(newrho), extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap=cm.hot)
# in this line, we'll add the B streamlines
plt.xlabel('x $[r_g]$', fontsize=16)
plt.ylabel('y $[r_g]$', fontsize=16)
plt.colorbar()
plt.title('Density at $t=950\;GM/c^3$', fontsize=20)
plt.show()
