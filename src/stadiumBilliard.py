#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:49:32 2017

@author: benjamin
"""

import math as m
import numpy as np
import verginiSaraceno as vs
import matplotlib.pyplot as pyp
import spectrumUtilities as su
import husimiFunctions as hf

def stadiumBoundaryDefinition(eps, k):
    # upper-right quater of the billiard
    # flat part
    mE = m.ceil(14 * eps * k / (2 * m.pi))
    xE = vs.baseAngles(0, eps, mE)
    yE = np.repeat(1, mE)
    wE = np.repeat((eps / mE), mE)
    # circular part
    mC = 4 * m.ceil(k)
    alpha = (m.pi / 2) - vs.baseAngles(0, m.pi / 2, mC)
    nx = np.cos(alpha)
    ny = np.sin(alpha)
    xC = eps + nx
    yC = ny
    wC = (m.pi / (2 * mC)) / (1 + eps * np.cos(alpha))
    # together
    x = np.concatenate((xE, xC))
    y = np.concatenate((yE, yC))
    w = np.concatenate((wE, wC))
    return x, y, w

def stadiumBoundaryDefinitionExtended(eps, k):
    # upper-right quater of the billiard
    # flat part
    mE = m.ceil(14 * eps * k / (2 * m.pi))
    xE = vs.baseAngles(0, eps, mE)
    yE = np.repeat(1, mE)
    nxE = np.repeat(0, mE)
    nyE = np.repeat(1, mE)
    sE = xE
    dsE = np.repeat(eps / mE, mE)
    # circular part
    mC = 4 * m.ceil(k)
    phi = vs.baseAngles(0, m.pi / 2, mC)
    alpha = (m.pi / 2) - phi 
    nxC = np.cos(alpha)
    nyC = np.sin(alpha)
    xC = eps + nxC
    yC = nyC
    sC = eps + phi
    dsC = np.repeat(m.pi / (2 * mC), mC)
    # together
    x = np.concatenate((xE, xC))
    y = np.concatenate((yE, yC))
    nx = np.concatenate((nxE, nxC))
    ny = np.concatenate((nyE, nyC))
    s = np.concatenate((sE, sC))
    ds = np.concatenate((dsE, dsC))
    return x, y, nx, ny, s, ds

def stadiumPlotEig(eps, k, vec, psi_fun, n = 100):
    ll = 1 + eps
    q = vs.baseAngles(0,ll,n)
    x = np.tile(q, n)
    y = np.repeat(q, n)
    psi = psi_fun(k, vec, x, y)
    Z = np.reshape(psi * psi, (n, n))
    X = np.reshape(x, (n, n))
    Y = np.reshape(y, (n, n))
    F = (X - eps) * (X - eps) + Y * Y
    C = np.logical_and(X > eps, F > 1)
    ind  = np.logical_or(C, Y > 1) 
    Z[ind] = 0
    Z = Z / np.max(Z)
    fig = pyp.figure()
    pyp.imshow(Z,cmap=pyp.get_cmap('gray'),interpolation='bilinear', 
               origin='lower',
               extent=(0,np.max(x),0,np.max(y)),vmin = 0,vmax=1)
    phs = np.linspace(0,0.5*m.pi,100)
    xb = eps + np.cos(phs)
    yb = np.sin(phs)
    xb = np.append(xb,0)
    yb = np.append(yb,1)
    pyp.plot(xb,yb)
    fig.show()
    

"""
asym_asym
"""

def stadiumSpecter_asym_asym(eps, k, dk, n):
    spek_fun = lambda x: stadiumEigvals_asym_asym(eps, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def stadiumEigvals_asym_asym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_asym_asym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def stadiumEig_asym_asym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_asym_asym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def stadiumSpectrumUnfold_asym_asym(eps, energies):
    A = 0.25 * m.pi + eps
    Scircle = 0.5 * m.pi + eps
    Sver = 1
    Shor = 1 + eps
    S = Scircle + Sver + Shor
    return su.unfoldWail(A, S, energies)

def stadiumPlotEig_asym_asym(eps, k, vec):
    stadiumPlotEig(eps, k, vec, vs.psi_pi2_asym_asym)
    
"""
sym_sym
"""

def stadiumSpecter_sym_sym(eps, k, dk, n):
    spek_fun = lambda x: stadiumEigvals_sym_sym(eps, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def stadiumEigvals_sym_sym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_sym_sym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def stadiumEig_sym_sym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_sym_sym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def stadiumSpectrumUnfold_sym_sym(eps, energies):
    A = 0.25 * m.pi + eps
    Scircle = 0.5 * m.pi + eps
    Sver = 1
    Shor = 1 + eps
    S = Scircle - Sver - Shor
    return su.unfoldWail(A, S, energies)

def stadiumPlotEig_sym_sym(eps, k, vec, n=100):
    stadiumPlotEig(eps, k, vec, vs.psi_pi2_sym_sym, n)
    
def stadiumHusimi_sym_sym(eps, k, vec):
    x, y, nx, ny, s, ds = stadiumBoundaryDefinitionExtended(eps, k)
    dpsi_x, dpsi_y = vs.grad_psi_pi2_sym_sym(k, vec, x, y)
    u = dpsi_x * nx + dpsi_y * ny
    s1 = np.concatenate((s, 2 * (eps + m.pi / 2) - np.flipud(s)))
    s2 = np.concatenate((-np.flipud(s), s1))
    ds1 = np.concatenate((ds, np.flipud(ds)))
    ds2 = np.concatenate((np.flipud(ds), ds1))
    u1 = np.concatenate((u, np.flipud(u)))
    u2 = np.concatenate((np.flipud(u), u1))
    nQ = 400
    nP = 400
    qs = vs.baseAngles(0, eps + m.pi / 2, nQ)
    ps = vs.baseAngles(0, 1, nP)
    H = hf.husimiOnGrid(k, s2, ds2, u2, qs, ps)
    return qs, ps, H

"""
asym_sym
"""

def stadiumSpecter_asym_sym(eps, k, dk, n):
    spek_fun = lambda x: stadiumEigvals_asym_sym(eps, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def stadiumEigvals_asym_sym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_asym_sym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def stadiumEig_asym_sym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_asym_sym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def stadiumSpectrumUnfold_asym_sym(eps, energies):
    A = 0.25 * m.pi + eps
    Scircle = 0.5 * m.pi + eps
    Sver = 1
    Shor = 1 + eps
    S = Scircle + Sver - Shor
    return su.unfoldWail(A, S, energies)

def stadiumPlotEig_asym_sym(eps, k, vec):
    stadiumPlotEig(eps, k, vec, vs.psi_pi2_asym_sym)
    
"""
asym_sym
"""

def stadiumSpecter_sym_asym(eps, k, dk, n):
    spek_fun = lambda x: stadiumEigvals_sym_asym(eps, x, dk)
    return  su.computeSpectrum(k, dk, n, spek_fun)

def stadiumEigvals_sym_asym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_sym_asym(n, k, w, x, y)
    d = vs.eigvals(k, dk, F, Fk)
    return d

def stadiumEig_sym_asym(eps, k, dk):
    x, y, w = stadiumBoundaryDefinition(eps, k)
    n = 3 * m.ceil(k / 4)
    F, Fk = vs.ffk_pi2_sym_asym(n, k, w, x, y)
    d, S = vs.eig(k, dk, F, Fk)
    return d, S

def stadiumSpectrumUnfold_sym_asym(eps, energies):
    A = 0.25 * m.pi + eps
    Scircle = 0.5 * m.pi + eps
    Sver = 1
    Shor = 1 + eps
    S = Scircle - Sver + Shor
    return su.unfoldWail(A, S, energies)

def stadiumPlotEig_sym_asym(eps, k, vec):
    stadiumPlotEig(eps, k, vec, vs.psi_pi2_sym_asym)


    

    
    
    