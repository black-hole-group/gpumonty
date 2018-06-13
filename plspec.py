#!/usr/bin/python
#
# first argument is dump number.
# you must edit the file to change which variable is plotted!
#
# import
import numpy as np
import matplotlib.pyplot as plt
#
# open spectrum file
data = np.loadtxt("spectrum.dat")
#data = np.loadtxt("100million_model24.dat")
tdata = np.transpose( data )
#
lw = tdata[0,:]	# log of photon energy in electron rest-mass units
#
nbins = 6
nspec = len(lw)
#
i = np.arange(0,nbins,1)
#
nLn = np.log10(tdata[1+i*7,:] + 1.e-30)
tauabs = tdata[2+i*7,:]
tauscatt = tdata[3+i*7,:]
x1av = tdata[4+i*7,:]
x2av = tdata[5+i*7,:]
x3av = tdata[6+i*7,:]
nscatt = tdata[7+i*7,:]
#
# normalize 
me = 9.1e-28
c = 3.e10
h = 6.626e-27
lw = lw + np.log10(me*c*c/h)   # convert to Hz from electron rest-mass energy
Lsol = 3.83e33  
nLn = nLn + np.log10(Lsol)  # convert to erg/s from Lsol
#
plt.step(lw, nLn[0], label = '$0^{\circ} - 15^{\circ}$', color = "red")
plt.step(lw, nLn[1], label = '$15^{\circ} - 30^{\circ}$', color = "orange")
plt.step(lw, nLn[2], label = '$30^{\circ} - 45^{\circ}$', color = "yellow")
plt.step(lw, nLn[3], label = '$45^{\circ} - 60^{\circ}$', color = "green")
plt.step(lw, nLn[4], label = '$60^{\circ} - 75^{\circ}$', color = "blue")
plt.step(lw, nLn[5], label = '$75^{\circ} - 90^{\circ}$', color = "black")
#
# in case I want the mean of the th_bins
#nLn_mean = np.mean(nLn, axis = 0)
#plt.step(lw, nLn_mean, label = 'mean')
# labels
plt.rc('text', usetex=True) 
plt.rc('font', size = 16)
plt.rc('font', family='times new roman')
plt.xlabel('$\log_{10}\\nu \;[{\\rm Hz}]$', weight='bold', fontsize=20)
plt.ylabel('$\log_{10}(\\nu L_\\nu) \;[{\\rm erg\\,\\,s}^{-1}]$', weight='bold', fontsize=20)
#
minlognu = 9
maxlognu = 23
#
eV = 1.602e-12
mum = 1.e-4 # 1 micron
ang = 1.e-8 # 1 angstrom

def plotenergy( freq, lab ):
	tmplnu = (np.log10(freq) - minlognu)/(maxlognu - minlognu)
	plt.figtext(tmplnu, 0.88, lab, rotation=-90, fontsize=15, va='top')
	return
"""
plotenergy(30.e3*eV/h, '30 keV')
plotenergy(1.e3*eV/h, '1 keV')
plotenergy(c/(5000.*ang), '5000 $\\AA$')
plotenergy(c/(2.*mum), '2 $\\mu$m')
plotenergy(c/(10.*mum), '10 $\\mu$m')
plotenergy(c/0.1, '1 mm')
"""
#
plt.xlim((minlognu, maxlognu))
plt.ylim((33.5-2.5, 33.5+3.5)) # Sgr A*
#plt.ylim((33.5+1.5, 33.5+11.5)) # M87
#

#
# choose model
sgra = 1
m87 = 0
#

# plotting the measured data at lambda = 1.3mm ~ 230Ghz
xx = np.log10(c*10/1.3)
if sgra:
	# normalization (2.4Jy)
	yy = xx + np.log10(2.4*10**23)
	# GHz
#	datax = np.array([235.6, 152.3, 151.0, 150.0, 106.3, 95.0, 95.0, 93.0, 43.3, 22.5, 14.9, 8.45, 4.85, 1.64, 1.44, 1.36, 0.33, 0.64, 1.15, 1.3, 1.5, 1.76, 5.0, 8.33, 15.3, 23.08, 42.86])
#	datax = np.log10(datax*10.e9)
	# Jy
#	datayjy = np.array([2.8, 2.9, 2.9, 3.1, 2.4, 2.1, 2.0, 1.9, 1.4, 1.1, 1.03, 0.72, 0.64, 0.55, 0.53, 0.53, 0.22, 0.45, 0.52, 0.52, 0.62, 0.63, 0.66, 0.69, 0.92, 1.06, 1.6])
	# nuFnu
#	datay = np.log10(10.e23*datayjy) + datax
	# error
#	errory = np.array([0.32, 0.09, 0.07, 0.13, 0.15, 0.11, 0.2, 0.26, 0.1, 0.05, 0.03, 0.01, 0.02, 0.05, 0.04, 0.04, 0.27, 0.22, 0.17, 0.12, 0.1, 0.08, 0.06, 0.04, 0.07, 0.06, 0.13])
	# plot
#	plt.errorbar(datax, datay, yerr = errory, fmt = 'o', markersize = 4, color = "black", label = "Data")
	xpldata = [9.93720930232558, 10.209302325581396, 10.146511627906976, 10.355813953488372, 10.606976744186046, 10.648837209302325, 10.962790697674418, 10.983720930232558, 10.983720930232558, 11.004651162790697, 11.172093023255814, 11.193023255813953, 11.36046511627907, 17.9953488372093]

	ypldata = [32.723755171921816, 33.116022257098024, 33.11610786132116, 33.324411470965906, 33.716707090883155, 33.71665002140106, 34.072050221144245, 34.157911256955344, 34.231530888857186, 34.3419318019689, 34.562562419746044, 34.62388357825653, 34.672735054929376, 32.4673705236125]

	plt.plot(xx, yy, marker = 'o', markersize = 8, color="red")
	plt.plot(xpldata, ypldata, 'o', markersize = 5, color = "black", label = "Data")

elif m87:
	# normalization (1.0Jy)
	yy = xx + np.log10(1.*10**23)

	
	xrh100_20, yrh100_20 = np.loadtxt("rh5_20.csv", delimiter = ",", unpack=True,usecols=(0,1))

	

	xpldata = [9.181167540213266, 9.369528284836436, 9.915525031628412, 10.319573468281224, 10.914476775709382, 10.982685703958069, 11.336743177299837, 14.341080787999278, 14.531935658774625, 14.676811856135913, 14.771697090186159, 14.50938008313754, 15.082161575998555, 14.939779504789445, 15.107211277787822, 15.011783842400146, 15.27551057292608, 17.352683896620277, 18.330923549611423, 22.332152539309597, 23.335767214892464, 24.36215434664739]

	ypldata = [38.27272727272727, 38.69090909090909, 39.14545454545454, 39.4, 39.654545454545456, 40.21818181818182, 40.85454545454545, 41.127272727272725, 41.127272727272725, 40.836363636363636, 40.92727272727272, 40.90909090909091, 40.872727272727275, 40.74545454545454, 40.67272727272727, 40.67272727272727, 40.45454545454545, 40.18181818181818, 40.163636363636364, 41.29090909090909, 41.018181818181816, 40.92727272727272]

	plt.plot(xx, yy, marker = 'o', markersize = 8, color="red")
	plt.plot(xpldata, ypldata, 'o', markersize = 5, color = "black", label = "Data")
	plt.step(xrh100_20, yrh100_20, lw = 3, color = "orange", label = "RH5(20)")

#
plt.legend(loc='lower center')#, bbox_to_anchor=(0.5, 0.5))
# plot on screen, where you can save to file after viewing plot
plt.show()
# or, uncomment to save directly...
#plt.savefig('tst.pdf')
