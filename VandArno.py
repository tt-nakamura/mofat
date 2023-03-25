import numpy as np
from numpy.linalg import norm

def VAorthog_(x, n=1, pol=None):
    """ Vandermonde-Arnordi Orthogonalization
    x = evaluation points (1d array of complex numbers)
    n = number of bases
    pol = poles (1d array of complex numbers)
    """
    m = len(x); r = 1/np.sqrt(m)
    if pol is not None: n = len(pol)
    Q = np.ones((n+1,m), dtype=np.complex)
    H = np.zeros((n+1,n), dtype=np.complex)
    for k in range(n):
        q = (x if pol is None else 1/(x-pol[k]))*Q[k]
        for j in range(k+1):
            H[j,k] = np.dot(q, Q[j])/m
            q -= H[j,k]*Q[j]
        H[k+1,k] = norm(q)*r
        Q[k+1] = q/H[k+1,k]
    return H
        
def VAeval_(x, H, pol=None, deriv=False):
    """
    x = evaluation points (complex numbers of any shape)
    H = Hessenberg matrix output by VAorthog
    pol = poles (1d array of complex numbers)
    deriv = evalutate derivatives or not
    """
    x = np.asarray(x)
    Q = np.ones(H.shape[:1] + x.shape, dtype=np.complex)
    for k in range(H.shape[1]):
        s = x if pol is None else 1/(x-pol[k])
        QH = np.einsum('i...,i', Q[:k], H[:k,k])
        Q[k+1] = (s*Q[k] - QH)/H[k+1,k]
    if not deriv:
        if pol is not None: Q = Q[1:]
        return np.moveaxis(Q,0,-1)

    D = np.zeros_like(Q)
    for k in range(H.shape[1]):
        if pol is None: s,t = x,1
        else: s,t = 1/(x-pol[k]), -1/(x-pol[k])**2
        DH = np.einsum('i...,i', D[:k], H[:k,k])
        D[k+1] = (s*D[k] - DH + t*Q[k])/H[k+1,k]
    if pol is not None: Q,D = Q[1:],D[1:]
    return (np.moveaxis(Q,0,-1),
            np.moveaxis(D,0,-1))

def VAorthog(x,n,pol):
    H = [VAorthog_(x,n)]
    for p in pol: H.append(VAorthog_(x,pol=p))
    return H

def VAeval(x, H, pol, deriv=False):
    Q = VAeval_(x, H[0], deriv=deriv)
    for h,p in zip(H[1:], pol):
        q = VAeval_(x,h,p,deriv)
        Q = np.concatenate((Q,q), axis=-1)
    return Q
