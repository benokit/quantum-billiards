import math as m
import numpy as np
import matplotlib.pyplot as pyp

def husimiOnGrid(k, s, ds, u, qs, ps):
    xs = [(q,p) for p in ps for q in qs]
    def f(x): 
        return husimiAtPoint(k, s, ds, u, x[0], x[1]) 
    hs = list(map(f, xs))
    return np.reshape(hs, (ps.size, qs.size)) 

def husimiAtPoint(k, s, ds, u, q, p):
    ss = s - q
    width = 4 / m.sqrt(k)
    indx = np.abs(ss) < width
    si = ss[indx]
    dsi = ds[indx]
    ui = u[indx]
    w = np.exp(-0.5 * k * si * si) * dsi
    cr = w * np.cos(k * p * si)
    ci = w * np.sin(k * p * si)
    hr = np.dot(cr, ui)
    hi = np.dot(ci, ui)
    return hr * hr + hi * hi

def entropy(H):
    H = H / np.sum(H)
    return -np.sum(H * np.log(H))

def entropyCover(entr, ncels):
    return m.exp(entr) / ncels

def correlation(H1, H2):
    f = 1 / (m.sqrt(np.sum(H1 * H1) * np.sum(H2 * H2)))
    return f * np.sum(H1 * H2) 

def plotHusimi(H, qmin, qmax, pmin, pmax):
    pyp.imshow(H,cmap=pyp.get_cmap('gray'),interpolation='bilinear',
               origin='lower',
               extent=(qmin,qmax,pmin,pmax))