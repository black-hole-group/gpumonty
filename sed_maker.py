import nmmn.sed
import matplotlib.pyplot as plt

s=nmmn.sed.SED()
s.grmonty('grmonty.spec')

plt.plot(s.lognu, s.ll)
plt.xlim(8,23)
plt.ylim(20,40)
plt.xlabel(r'$\nu$', fontsize=16)
plt.ylabel(r'$\nu L_{\nu}$', fontsize=16)
plt.title(r'$N_s = 10^3$', fontsize=20)
plt.show()
