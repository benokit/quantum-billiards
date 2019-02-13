import math as m

import matplotlib.pyplot as pyp
import numpy as np
import scipy

from . import tools


# brody distribution
def brodyP(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return (beta + 1) * a * np.power(s, beta) * np.exp(-a * np.power(s, beta + 1))


# cumulative brody distribution
def brodyW(beta, s):
    a = m.gamma ((beta + 2) / (beta + 1)) ** (beta + 1)
    return 1 - np.exp(-a * np.power(s, beta + 1))

# U distribution
def distU(ws):
    return (2 / m.pi) * np.arccos(np.sqrt(1 - ws))

# U brody distribution
def brodyU(beta, s):
    return distU(brodyW(beta, s))

# brody gap probability
def brodyE(beta, s):
    a = m.gamma((beta + 2)/(beta + 1)) ** (beta + 1)
    x = a * np.power(s, beta + 1)
    return scipy.special.gammaincc(1/(beta + 1), x)

# Berry - Robnik brody gap probability 
def brodyBRE(rho, beta, s):
    return np.exp(-rho * s) * brodyE(beta, (1 - rho) * s)

# Berry - Robnik brody kumulative distribution
def brodyBRW(rho, beta, s):
    a = (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) - rho * brodyE(beta, (1 - rho) * s)
    return a * np.exp(-rho * s) + 1

# Berry - Robnik brody porbability density
def brodyBRP(rho, beta, s):
    a = (rho ** 2) * brodyE(beta, (1 - rho) * s) - 2 * rho * (1 - rho) * (brodyW(beta, (1 - rho) * s) - 1) + ((1 - rho) ** 2) * brodyP(beta, (1 - rho) * s) 
    return a * np.exp(-rho * s)

# Berry - Robnik brody U distribution
def brodyBRU(rho, beta, s):
    (2 / m.pi) * np.arccos(np.sqrt(1 - brodyBRW(rho, beta, s)))

# find the best fitting beta 
def brodyFit(s):
    z = s / np.mean(s)
    Wz = tools.ecdf(z)
    x = np.linspace(0, 3, 1000)
    y = Wz(x)
    funk = lambda beta: np.sum(np.power(brodyW(beta, x) - y, 2))
    res = scipy.optimize.minimize(funk, [0], bounds = [(0,1)], method = 'L-BFGS-B')
    return res.x[0]

# find the best fitting beta and rho
# or best fitting beta while rho is fixed with fixed = T
def brodyBRFit(s, rho = 0, fixedRho = True):
    z = s / np.mean(s)
    Wz = tools.ecdf(z)
    x = np.linspace(0, 3, 1000)
    y = Wz(x)
    if (fixedRho):
        funk = lambda beta: np.sum(np.power(brodyBRW(rho, beta, x) - y, 2))
        res = scipy.optimize.minimize(funk, [0], bounds = [(0,1)], method = 'L-BFGS-B')
        return rho, res.x[0]
    else:
        funk = lambda par: np.sum(np.power(brodyBRW(par[0], par[1], x) - y, 2))
        res = scipy.optimize.minimize(funk, [0.5, 0.5], bounds = [(0,1), (0,1)], method = 'L-BFGS-B')
        return res.x[0], res.x[1]
    
# ploting P
def plotBrodyFitP(s):
    z = s / np.mean(s)
    beta = brodyFit(z)
    pyp.hist(z, histtype='step', normed=True, bins=30, range=(0,6))
    x = np.linspace(0, 6, 1000)
    pyp.plot(x, brodyP(beta, x))
    pyp.show()
    
# ploting W
def plotBrodyFitW(s, wsmin = 0, wsmax = 3):
    z = s / np.mean(s)
    beta = brodyFit(z)
    Wz = tools.ecdf(z)
    x = np.linspace(wsmin, wsmax, 1000)
    y = Wz(x)
    pyp.plot(x, y)
    pyp.plot(x, brodyW(beta, x))
    pyp.show()
    
# ploting W
def plotBrodyFitW_loglog(s, wsmin = 0, wsmax = 1):
    z = s / np.mean(s)
    beta = brodyFit(z)
    Wz = tools.ecdf(z)
    x = np.linspace(wsmin, wsmax, 1000)
    y = Wz(x)
    pyp.plot(np.log10(x), np.log10(y))
    pyp.plot(np.log10(x), np.log10(brodyW(beta, x)))
    pyp.show()
    
# ploting U
def plotBrodyFitU(s, wsmin = 0, wsmax = 3):
    z = s / np.mean(s)
    beta = brodyFit(z)
    Wz = tools.ecdf(z)
    Uf = lambda zz: distU(Wz(zz))
    Zmin = np.min(z)
    Zmax = np.max(z)
    Umin = Uf(Zmin)
    Umax = Uf(Zmax)
    Uerr = 1 / (m.pi * m.sqrt(z.size))
    zz = np.linspace(Zmin, Zmax, 1000)
    ux = np.linspace(Umin, Umax, 200)
    zx = scipy.interpolate.spline(brodyU(beta, zz), zz, ux)
    y = Uf(zx) - brodyU(beta,zx)
    pyp.plot(ux, y)
    pyp.fill_between(ux, y + Uerr, y - Uerr)
    pyp.show()
