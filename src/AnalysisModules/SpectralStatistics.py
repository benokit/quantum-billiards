import numpy as np
from scipy import special
import sys
sys.path.append("..")
from ..spectral_functions import compute_sff, compute_number_variance


def weight_function(E,E0,sigma):
    return np.sqrt(2/(sigma*np.sqrt(np.pi))*np.exp(-4/(sigma**2)*np.power((E - E0), 2.)))

def SFF(E, a=1, b=1, min_t = 0, max_t = 2 ):
    "Computes spectral form factor of unfolded spectra in time units of Heisenberg time."
    E0 = np.mean(E)
    W = E[-1]-E[0]
    dt = round(a/(W),7)
    print("dt = %s" %dt)
    print("E0 = %s" %E0)
    print("W = %s" %W)
    tim = np.arange(min_t ,max_t, dt)
    if b == "inf":
        weights = np.ones(len(E))
    else:
        weights = weight_function(E,E0, sigma = W/2*b)
    tim, sff = compute_sff(E,tim,weights)
    return tim, sff

def NV(E, min_l=0, max_l = 20, grid = 50):
    "Computes number variance of unfolded spectra."
    Ls = np.linspace(min_l, max_l, grid)
    return compute_number_variance(Ls, E)

def NV_GOE(l):
    pl = np.pi*l
    pl2 = 2*np.pi*l 
    si1, ci1 = special.sici(pl) 
    si2, ci2 = special.sici(pl2) 
    
    return 2/np.pi**2*(np.log(pl2) + np.euler_gamma + 1 +0.5*si1**2 - np.pi/2*si1 -np.cos(pl2) - ci2 + np.pi**2*l*(1-2/np.pi*si2))