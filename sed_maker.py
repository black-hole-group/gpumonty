import nmmn.sed
import matplotlib.pyplot as plt

s=nmmn.sed.SED()
#s.grmonty('grmonty.spec')
s.grmonty('spectrum.dat')

plt.plot(s.lognu, s.ll)
plt.xlim(8,23)
plt.ylim(20,40)
plt.xlabel(r'$\log\;\nu\;(\mathrm{Hz})$', fontsize=16)
plt.ylabel(r'$\nu L_{\nu}\;(\mathrm{erg\;s}^{-1})$', fontsize=16)
plt.title(r'$N_s = 10^5$', fontsize=20)
plt.show()
