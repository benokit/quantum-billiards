import math as m

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy import optimize
from scipy import interpolate


# U distribution
def distU(ws):
    return (2 / m.pi) * np.arccos(np.sqrt(1 - ws))

#define cumulative density function
def ecdf(z):
    zz = np.sort(z)
    n = 1 + zz.size
    f = lambda x: np.count_nonzero(zz <= x) / n
    return np.vectorize(f)

def P(s, smin = 0, smax = 4, grid = 50):
    h, bins = np.histogram(s, bins=grid, range=(smin,smax), density=True)
    return h, (bins[1:] + bins[:-1])/2

def W(s, smin = 0, smax = 4, grid = 200):
    z = s #/ np.mean(s)
    Wz = ecdf(z)
    x = np.linspace(smin, smax, grid)
    y = Wz(x)
    return x, y

def U(s, smin = 0, smax = 4, grid = 200):
    z = s #/ np.mean(s)
    Wz = ecdf(z)
    x = np.linspace(smin, smax, grid)
    Uf = lambda zz: distU(Wz(zz))
    y = Uf(x)
    return x, y

# brody distribution
def brodyP(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return (beta + 1) * a * np.power(s, beta) * np.exp(-a * np.power(s, beta + 1))


# cumulative brody distribution
def brodyW(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return 1 - np.exp(-a * np.power(s, beta + 1))



# U brody distribution
def brodyU(beta, s):
    return distU(brodyW(beta, s))

# brody gap probability
def brodyE(beta, s):
    a = m.gamma((beta + 2)/(beta + 1)) ** (beta + 1)
    x = a * np.power(s, beta + 1)
    return special.gammaincc(1/(beta + 1), x)

# Berry - Robnik brody gap probability 
def brodyBRE(rho, beta, s):
    return np.exp(-rho * s) * brodyE(beta, (1 - rho) * s)

# Berry - Robnik brody cumulative distribution
def brodyBRW(rho, beta, s):
    a = (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) - rho * brodyE(beta, (1 - rho) * s)
    return a * np.exp(-rho * s) + 1

# Berry - Robnik brody probability density
def brodyBRP(rho, beta, s):
    a = (rho ** 2) * brodyE(beta, (1 - rho) * s) - 2 * rho * (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) + ((1 - rho) ** 2) * brodyP(beta, (1 - rho) * s) 
    return a * np.exp(-rho * s)

# Berry - Robnik brody U distribution
def brodyBRU(rho, beta, s):
    (2 / m.pi) * np.arccos(np.sqrt(1 - brodyBRW(rho, beta, s)))

# find the best fitting beta 
def brodyFit(s):
    z = s / np.mean(s)
    Wz = ecdf(z)
    x = np.linspace(0, 3, 1000)
    y = Wz(x)
    funk = lambda beta: np.sum(np.power(brodyW(beta, x) - y, 2))
    res = optimize.minimize(funk, [0], bounds = [(0,1)], method = 'L-BFGS-B')
    return res.x[0]

# find the best fitting beta and rho
# or best fitting beta while rho is fixed with fixed = T
def brodyBRFit(s, rho = 0, fixedRho = True):
    z = s / np.mean(s)
    Wz = ecdf(z)
    x = np.linspace(0, 3, 1000)
    y = Wz(x)
    if (fixedRho):
        funk = lambda beta: np.sum(np.power(brodyBRW(rho, beta, x) - y, 2))
        res = optimize.minimize(funk, [0], bounds = [(0,1)], method = 'L-BFGS-B')
        return rho, res.x[0]
    else:
        funk = lambda par: np.sum(np.power(brodyBRW(par[0], par[1], x) - y, 2))
        res = optimize.minimize(funk, [0.5, 0.5], bounds = [(0,1), (0,1)], method = 'L-BFGS-B')
        return res.x[0], res.x[1]

#Wigner-Dyson distributions
def wdP(s):
    return brodyP(1, s)

def wdW(s):
    return brodyW(1, s)

def wdU(s):
    return brodyU(1, s)

#semi-poisson distributions
def spP(s):
    return 4*s*np.exp(-2*s)

#cumulative 
def spW(s):
    return 1 - (2*s+1)*np.exp(-2*s)

# U 
def spU(s):
    return distU(spW(s))