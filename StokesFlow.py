# reference:
#   P. D. Brubeck and L. N. Trefethen
#    "Lighting Stokes Solver"

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq,norm
from matplotlib.path import Path
from VandArno import VAorthog,VAeval

class StokesFlow:
    def __init__(self, c, n, nc, Psi, W, d=None,
                 fix=None, bound=None, pol_dir=None,
                 NPTS=5, trange=15, NPTS_ratio=None,
                 scale=None, sigma=4, weight='dist'):
        """
        c = corner vertices (1d array of complex numbers)
        n = degree of polynomial basis (scalar integer)
        nc = number of clustring points around corners
             (scalar or 1d arrray of integers)
        Psi = boundary value of stream function along edges
              (1d array of real numbers or nans)
        W = boundary value of velocity (1d array of real numbers)
        d = direction of velocity on boundaries
              (1d array of non-zero complex numbers)
        fix = points at which to fix consts in f,g
        bound = boundary curves
        pol_dir = direction of poles (1d array of complex numbers)
        NPTS = number of evaluation points per basis
        trange = range of tanh
        NPTS_ratio = proportinal to number of sample points on edges
                     (1d array of real numbers)
        scale = length of poles clustering from corners
        sigma = density of poles around corners
        weight = 'd' if weighting by distance to nearest corner
                 else equal-norm weighting
        """
        K = len(c)
        c = np.asarray(c) # vertices
        e = np.diff(np.r_[c,c[0]]) # edges
        a = np.unwrap(np.angle(e))

        if d is None: d = e/np.abs(e)
        if np.isscalar(nc): nc = np.full(K,nc)

        # set number of sample points along edges
        m = (n + np.nansum(nc))*2*NPTS
        if NPTS_ratio is None:
            m = np.full(K, m//K).astype(np.int)
        else:
            m *= np.asarray(NPTS_ratio)
            m = (m/np.sum(NPTS_ratio)).astype(np.int)

        # collect sample points
        z,s = [],[]
        b = [None]*K if bound is None else bound
        for i in range(K):
            t = np.linspace(-trange, trange, m[i])
            t = (1 + np.tanh(t))/2
            w = b[i](t) if callable(b[i]) else c[i] + t*e[i]
            z = np.r_[z,w]
            s.append(t)

        # collect poles
        pol = []
        t = np.exp(1j*(a + np.roll(a,1))/2)/1j # exterior bisector
        t[0] = -t[0]
        p = [np.nan]*K if pol_dir is None else pol_dir
        L = np.max(np.abs(e)) if scale is None else scale
        for i in range(K):
            if np.isnan(nc[i]): continue
            if not np.isnan(p[i]): t[i] = p[i]/np.abs(p[i])
            dk = np.arange(nc[i],0,-1)
            dk = np.exp(sigma*(np.sqrt(dk) - np.sqrt(nc[i])))
            pol.append(c[i] + L*dk*t[i])

        H = VAorthog(z,n,pol)
        phi,phi_p = VAeval(z,H,pol,True)
        N = phi.shape[1]

        zz = z.reshape(-1,1)
        z_phi = np.conj(zz)*phi
        g1 = np.conj(zz)*phi_p
        g1,g2 = g1-phi,g1+phi

        # Psi = Im(conj(z)*f + g)
        # u = Re(conj(z)*f' - f + g')
        # v = -Im(conj(z)*f' + f + g')
        psi = np.c_[np.imag(z_phi),np.real(z_phi), # Re f, Im f
                    np.imag(phi),  np.real(phi)] # Re g, Im g
        u = np.c_[np.real(g1),   -np.imag(g1),
                  np.real(phi_p),-np.imag(phi_p)]
        v = -np.c_[np.imag(g2),   np.real(g2),
                   np.imag(phi_p),np.real(phi_p)]
        A1 = np.empty_like(psi)
        A2 = np.empty_like(psi)
        b1 = np.empty_like(z)
        b2 = np.empty_like(z)

        # set boundary conditions
        m = np.r_[0, np.cumsum(m)]
        for i in range(K):
            j = np.s_[m[i]:m[i+1]]
            w = W[i](s[i]) if callable(W[i]) else W[i]
            t = d[i](s[i]) if callable(d[i]) else (
                e[i]/np.abs(e[i]) if np.isnan(d[i]) else d[i])
            p = Psi[i](s[i]) if callable(Psi[i]) else Psi[i]
            if np.isscalar(p) and np.isnan(p):# Neumann
                w *= t
                A1[j],A2[j] = u[j],v[j]
                b1[j],b2[j] = np.real(w), np.imag(w)
            else:# Dirichlet
                t = np.angle(t)
                A1[j] = psi[j]
                A2[j] = np.cos(t)*u[j] + np.sin(t)*v[j]
                b1[j],b2[j] = p,w

        A = np.vstack((A1,A2))
        b = np.hstack((b1,b2))

        # row weighting
        if weight[0].lower() == 'd':# distance to nearest corner
            w = np.min(np.abs(zz-c), axis=1)
            w = np.r_[w,w]
        else:# equal-norm weighting
            w = 1/norm(A, axis=1)
        A *= w.reshape(-1,1)
        b *= w

        # fix constants in f and g
        if fix is not None:
            p = VAeval(fix[0],H,pol)
            q = VAeval(fix[1],H,pol)
            o = np.zeros(N)
            A = np.vstack((A,
                           np.r_[np.real(p),-np.imag(p),o,o],
                           np.r_[np.imag(p), np.real(p),o,o],
                           np.r_[o,o,np.real(p),-np.imag(p)],
                           np.r_[np.real(q),-np.imag(q),o,o]))
            b = np.hstack((b, [0]*4))

        # solve normal equation by least-square method
        x = lstsq(A,b,rcond=-1)[0]
        f,g = x[:2*N],x[2*N:]
        f = f[:N] + 1j*f[N:]
        g = g[:N] + 1j*g[N:]

        self.f = f
        self.g = g
        self.H = H
        self.pol = pol
        self.vertex = c
        self.bound = bound

    def stream_func(self, z):
        phi =  VAeval(z, self.H, self.pol)
        f,g = np.dot(phi, self.f), np.dot(phi, self.g)
        psi = np.imag(np.conj(z)*f + g)
        if np.isscalar(psi): return psi
        D = self.isinside(z)
        psi[~D] = np.nan
        return psi

    def velocity(self, z):
        phi,phi_p = VAeval(z, self.H, self.pol, True)
        f = np.dot(phi, self.f)
        df = np.dot(phi_p, self.f)
        dg = np.dot(phi_p, self.g)
        w = z*np.conj(df) - f + np.conj(dg)
        if np.isscalar(w): return w
        D = self.isinside(z)
        w[~D] = np.nan
        return w

    def isinside(self, z):
        p = Path([(z.real, z.imag) for z in self.boundary()])
        D = np.dstack((np.real(z), np.imag(z))).reshape(-1,2)
        e = p.get_extents()
        r = 0.004*(e.xmax - e.xmin)
        D = p.contains_points(D, radius=r)
        return D.reshape(z.shape)

    def boundary(self, N=100):
        if self.bound is None:
            return self.vertex
        B = []
        t = np.linspace(0,1,N,endpoint=False)
        for i,b in enumerate(self.bound):
            if callable(b): B = np.r_[B,b(t)]
            else: B = np.r_[B, self.vertex[i]]
        return B

    def plot_boundary(self, *a, **k):
        b = np.r_[self.boundary(), self.vertex[0]]
        plt.plot(np.real(b), np.imag(b), *a, **k)

    def plot_poles(self, *a, **k):
        for p in self.pol:
            plt.plot(np.real(p), np.imag(p), *a, **k)
