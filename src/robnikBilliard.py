#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:49:32 2017

@author: benjamin
"""

import math as m
import numpy as np
import verginiSaraceno as vs
import spectrumUtilities as su
import scipy.integrate as integrate

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
    ny = (np.sin(alpha) + 2 * lam * np.sin(2 * alpha)) / dsdalpha
    q = 1 + 4 * lam * lam + 4 * lam * np.cos(alpha)
    z = 1 + 2 * lam * lam + 3 * lam * np.cos(alpha)
    w = dalpha * q / z
    return x, y, w

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


    

    
    
    