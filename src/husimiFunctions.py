import math as m
import numpy as np
import matplotlib.pyplot as plt

def husimiAtPoint(k, s, ds, u, q, p):
    """Calculates the Poincare-Husimi function at point (q,p) in the quantum phase space
       - k is the wavenumber of the eigen state
       - s is an array of points on the boundary
       - ds is an array of boundary integration weights
       - u is an array of boundary function values
    """
    ss = s - q
    width = 4 / m.sqrt(k)
    indx = np.abs(ss) < width
    si = ss[indx]
    dsi = ds[indx]
    ui = u[indx]
    w = np.sqrt(np.sqrt(k/np.pi)) * np.exp(-0.5 * k * si * si) * dsi
    cr = w * np.cos(k * p * si) #coherent state real part
    ci = w * np.sin(k * p * si) #coherent state imaginary part
    h = np.dot(cr - 1j*ci, ui) # Husimi integral minus because of conjugation
    hr = h.real
    hi = h.imag
    return (hr * hr + hi * hi)/(2*np.pi*k)

def husimiOnGrid(k, s, ds, u, qs, ps):
    """Evaluates the Poicare-Husimi function on the grid given by the arrays qs and ps
       - qs is a 1d array of points on the billiard boundary 
       - ps is a 1d array of points in the cannonical momentum
       - k is the wavenumber of the eigen state
       - s is an array of points on the boundary
       - ds is an array of boundary integration weights
       - u is an array of boundary function values
    """
    xs = [(q,p) for p in ps for q in qs]
    def f(x): 
        return husimiAtPoint(k, s, ds, u, x[0], x[1]) 
    hs = list(map(f, xs))
    return np.reshape(hs, (ps.size, qs.size)) 

def entropy(H):
    H = H / np.sum(H)
    return -np.sum(H * np.log(H))

def entropyCover(entr, ncels):
    return m.exp(entr) / ncels

def correlation(H1, H2):
    f = 1 / (m.sqrt(np.sum(H1 * H1) * np.sum(H2 * H2)))
    return f * np.sum(H1 * H2) 

