import numpy as np
import matplotlib.pyplot as plt
from StokesFlow import StokesFlow

c = [1+1j, -1+1j, -1, -1j, 1-1j]
n,nc = 40,30
psi,w = [0,0,0,0,0],[1,0,0,0,0]
b = [np.nan, np.nan,
     lambda t: -1-1j+np.exp(1j*np.pi/2*(1-t)),
     np.nan, np.nan]
pd = [1+1j, -1+1j, -1-1j, -1-1j, 1-1j]
m = StokesFlow(c, n, nc, psi, w, fix=(0,1), bound=b, pol_dir=pd)

x,y = np.mgrid[-1:1:100j,-1:1:100j]
z = x + 1j*y
psi = m.stream_func(z)
w = np.abs(m.velocity(z))
wmin,wmax = np.nanmin(w), np.nanmax(w)
pmin,pmax = np.nanmin(psi), np.nanmax(psi)
fac = pmin/pmax
lev1 = np.r_[wmin:wmax:100j]
lev2 = pmin + (pmax-pmin)*np.r_[0.1:0.9:5j]
lev3 = (lev2*fac)[::-1]

plt.figure(figsize=(5,4.8))
plt.axis('equal')
plt.contourf(x,y,w,cmap='jet',levels=lev1)
plt.contour(x,y,psi,colors='k',linestyles='solid',levels=lev2)
plt.contour(x,y,psi,colors='y',linestyles='solid',levels=lev3)
plt.box('off')
plt.xticks([])
plt.yticks([])
m.plot_boundary('k', lw=2)
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
