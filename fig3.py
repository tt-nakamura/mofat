import numpy as np
import matplotlib.pyplot as plt
from StokesFlow import StokesFlow

a = np.deg2rad(14.25)
X1,X2 = -0.3, -0.3 + np.cos(a)
Y = np.sin(a)
c = [X2, X1 + 1j*Y, X1 - 1j*Y]
n,nc = 24,24
psi,w = [0,0,0],[0,1,0]
m = StokesFlow(c, n, nc, psi, w, fix=(0,X2), scale=2)

x,y = np.mgrid[X1:X2:100j, -Y:Y:100j]
z = x + 1j*y
psi = m.stream_func(z)
w = np.abs(m.velocity(z))
wmin,wmax = np.nanmin(w), np.nanmax(w)
pmin,pmax = np.nanmin(psi), np.nanmax(psi)
fac = pmin/pmax
lev1 = np.r_[wmin:wmax:100j]
lev2 = pmin + (pmax-pmin)*np.r_[0.2:0.8:4j]
lev3 = (lev2*fac)[::-1]
lev4 = (lev3*fac)[::-1]

z /= 1j # rotate by -90
x,y = np.real(z),np.imag(z)
b = np.r_[c,c[0]]/1j

plt.figure(figsize=(5,5*1.97))
plt.axis('equal')
plt.contourf(x,y,w,cmap='jet',levels=lev1)
plt.contour(x,y,psi,colors='k',linestyles='solid',levels=lev2)
plt.contour(x,y,psi,colors='y',linestyles='solid',levels=lev3)
plt.contour(x,y,psi,colors='y',linestyles='solid',levels=lev4)
plt.plot(np.real(b), np.imag(b), 'k', lw=2)
plt.box('off')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
