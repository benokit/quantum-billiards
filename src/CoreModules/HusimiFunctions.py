import math as m
import numpy as np
import sys
sys.path.append("..")
from .. import husimi_cy  

def husimiOnGrid(k, s, ds, u, qs, ps):
    H = husimi_cy.husimiOnGrid_py(k, s, ds, u, qs, ps)
    return H 

def entropy(H):
    return -np.sum(H * np.log(H))

def entropyCover(entr, ncels):
    return m.exp(entr) / ncels

def IPR(H, ncels):
    return 1/ncels/np.sum(H**2)

def correlation(H1, H2):
    f = 1 / (m.sqrt(np.sum(H1 * H1) * np.sum(H2 * H2)))
    return f * np.sum(H1 * H2) 

def Renyi_measure(H, a):
    row, col = H.shape
    ncels = row*col
    #print(ncels)
    if a==1:
        l = np.exp(-np.sum(H * np.log(H)))/ncels
    else:
        l = 1/(ncels)*(np.sum(H**a))**(1/(1-a))
    return l