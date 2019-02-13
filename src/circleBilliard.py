import numpy as np
import math as m
from . import verginiSaraceno as vs
from . import matplotlib.pyplot as pyp

def circle_eig(k, dk):
    """
        eigenvalues and eigenvectors of a circle billiard 
        on the interval [k-dk,k+dk]
    """
    mm = 14 * m.ceil(k)   
    phi = vs.baseAngles(0, 2 * m.pi, mm)
    x = np.cos(phi)
    y = np.sin(phi)
    w = np.repeat(2 * m.pi / mm, mm)
    n = m.ceil(2 * k / 2)
    F, Fk = vs.ffk_pi_sym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def circle_plot_eig(k, vec):
    """
    """
    n = 400
    q = np.linspace(-1,1,n)
    x = np.tile(q, n)
    y = np.repeat(q, n)
    psi = vs.psi_pi_sym(k, vec, x, y)
    Z = np.reshape(psi * psi, (n, n))
    X = np.reshape(x, (n, n))
    Y = np.reshape(y, (n, n))
    F = X * X + Y * Y
    ind  = F > 1
    Z[ind] = 0
    pyp.contourf(X, Y, Z, cmp = pyp.cm.gray)
    
    
    
    
    