import math as m
import numpy as np
from . import verginiSaraceno as vs
from . import spectrumUtilities as su
import scipy.integrate as integrate
import husimiFunctions as hf

def robnikBoundaryDefinition(lam, k, mC = None):
    if (mC == None):
        mC = 8 * m.ceil(k)
    dalpha = m.pi / mC
    alpha = vs.baseAngles(0, m.pi, mC)
    x = np.cos(alpha) + lam * np.cos(2 * alpha)
    y = np.sin(alpha) + lam * np.sin(2 * alpha)
    q = 1 + 4 * lam * lam + 4 * lam * np.cos(alpha)
    z = 1 + 2 * lam * lam + 3 * lam * np.cos(alpha)
    w = dalpha * q / z
    return x, y, w

def robnikBoundaryDefinitionExtended(lam, k, mC = None):
    if (mC == None):
        mC = 8 * m.ceil(k)
    dalpha = m.pi / mC
    alpha = vs.baseAngles(0, m.pi, mC)
    x = np.cos(alpha) + lam * np.cos(2 * alpha)
    y = np.sin(alpha) + lam * np.sin(2 * alpha)
    dsdalpha = np.sqrt(1 + 4 * lam * (lam + np.cos(alpha)))
    nx = (np.cos(alpha) + 2 * lam * np.cos(2 * alpha)) / dsdalpha
    ny = (np.sin(alpha) + 2 * lam * np.cos(2 * alpha)) / dsdalpha
    ds = dsdalpha * dalpha
    s = np.cumsum(ds) - 0.5 * ds
    return x, y, nx, ny, s, ds

"""
asym
"""

def robnikSpecter_asym(lam, k, dk, n):
    spek_fun = lambda x: robnikEigvals_asym(lam, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def robnikEigvals_asym(lam, k, dk, n = None):
    if (n == None):
        n = 200 + m.ceil(0.5 * (1 + lam * lam) * k)
    x, y, w = robnikBoundaryDefinition(lam, k)
    F, Fk = vs.ffk_pi_asym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def robnikEig_asym(lam, k, dk, n = None):
    if (n == None):
        n = 200 + m.ceil(0.5 * (1 + lam * lam) * k)
    x, y, w = robnikBoundaryDefinition(lam, k)
    F, Fk = vs.ffk_pi_asym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def robnikSpectrumUnfold_asym(lam, energies):
    A = 0.5 * (1 + 2 * lam * lam) * m.pi
    Scurve = integrate.quad(lambda x: m.sqrt(1 + 4 * lam * (lam + m.cos(x))), 0, m.pi)[0]
    Shor = 2.0
    S = Scurve + Shor
    return su.unfoldWail(A, S, energies)
    
"""
sym
"""

def robnikSpecter_sym(lam, k, dk, n):
    spek_fun = lambda x: robnikEigvals_sym(lam, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def robnikEigvals_sym(lam, k, dk, n = None):
    if (n == None):
        n = 200 + m.ceil(0.5 * (1 + lam * lam) * k)
    x, y, w = robnikBoundaryDefinition(lam, k)
    F, Fk = vs.ffk_pi_sym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def robnikEig_sym(lam, k, dk, n = None):
    if (n == None):
        n = 200 + m.ceil(0.5 * (1 + lam * lam) * k)
    x, y, w = robnikBoundaryDefinition(lam, k)
    F, Fk = vs.ffk_pi_sym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def robnikSpectrumUnfold_sym(lam, energies):
    A = 0.5 * (1 + 2 * lam * lam) * m.pi
    Scurve = integrate.quad(lambda x: m.sqrt(1 + 4 * lam * (lam + m.cos(x))), 0, m.pi)[0]
    Shor = 2.0
    S = Scurve - Shor
    return su.unfoldWail(A, S, energies)

def robnikHusimi_sym(lam, k, vec):
    x, y, nx, ny, s, ds = robnikBoundaryDefinitionExtended(lam, k)
    dpsi_x, dpsi_y = vs.grad_psi_pi_sym(k, vec, x, y)
    u = dpsi_x * nx + dpsi_y * ny
    Scurve = integrate.quad(lambda z: m.sqrt(1 + 4 * lam * (lam + m.cos(z))), 0, m.pi)[0]
    s1 = np.concatenate((s, 2 * Scurve - np.flipud(s)))
    s2 = np.concatenate((-np.flipud(s), s1))
    ds1 = np.concatenate((ds, np.flipud(ds)))
    ds2 = np.concatenate((np.flipud(ds), ds1))
    u1 = np.concatenate((u, np.flipud(u)))
    u2 = np.concatenate((np.flipud(u), u1))
    nQ = 400
    nP = 400
    qs = vs.baseAngles(0, Scurve, nQ)
    ps = vs.baseAngles(0, 1, nP)
    H = hf.husimiOnGrid(k, s2, ds2, u2, qs, ps)
    return qs, ps, H


    

    
    
    