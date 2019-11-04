#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:14:18 2017

@author: benjamin
"""

import pandas as pa
import matplotlib.pyplot as pyp
from scipy.optimize import curve_fit

betaDataPath = '/home/benjamin/Research/QuantumChaos/RobnikQuantum/report/tabela.csv'

alphaDataPath = '/home/benjamin/Research/QuantumChaos/RobnikQuantum/report/transportTimesRobnik.csv'

def plotBetaVsAlpha(ntcol = 'Nt5', percent = 50, s = 0.11, marksize=3, fontsize=8):
    dfb = pa.read_csv(betaDataPath)
    dfb = dfb[dfb['kmin'].isin([2000,4000])]
    dfa = pa.read_csv(alphaDataPath)
    df = dfb.join(dfa.set_index('lambda'), on='lambda')
    x = (df['kmin'] + df['kmax'])/df[ntcol]
    y = df['beta']
    pyp.plot(x,y,'bo', markersize=marksize)
    ii = x.values.argsort()
    xx = x.values[ii]
    yy = y.values[ii]
    popt, pcov = curve_fit(model, xx, yy, bounds=(0,10))
    pyp.plot(xx, model(xx, 0.98, s), 'r-', label='fit')
    pyp.ylabel(r'$\beta$')
    pyp.xlabel(r'$\alpha({0}\%)$'.format(percent))
    pyp.title(r'Lambda billiard. Model: $A = {0}$, $s = {1:.2f}$'.format(0.98, s), fontsize=fontsize)
    return popt

def plotAll():
    pyp.figure(1)
    pyp.subplot(221)
    plotBetaVsAlpha('Nt9', 90, 0.55)
    pyp.subplot(222)
    plotBetaVsAlpha('Nt8', 80, 0.26)
    pyp.subplot(223)
    plotBetaVsAlpha('Nt7', 70, 0.15)
    pyp.subplot(224)
    plotBetaVsAlpha('Nt5', 50, 0.06)
    pyp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyp.savefig('/home/benjamin/Documents/Temp/lambdaBilliardBvsA.eps', format='eps', dpi=1000)
    
def plotCoverVsAlpha(ntcol = 'Nt5', percent = 50, s = 0.11, marksize=3, fontsize=8):
    dfb = pa.read_csv(betaDataPath)
    dfb = dfb[dfb['kmin'].isin([2000,4000])]
    dfa = pa.read_csv(alphaDataPath)
    df = dfb.join(dfa.set_index('lambda'), on='lambda')
    x = (df['kmin'] + df['kmax'])/df[ntcol]
    y = df['corr']
    a = max(y)
    pyp.plot(x,y,'bo', markersize=marksize)
    ii = x.values.argsort()
    xx = x.values[ii]
    yy = y.values[ii]
    b, pcov = curve_fit(lambda x,b: model(x,a,b), xx, yy, bounds=(0,10))
    s = b[0] * 0.75
    pyp.plot(xx, model(xx, a, s), 'r-', label='fit')
    pyp.ylabel(r'$C$')
    pyp.xlabel(r'$\alpha({0}\%)$'.format(percent))
    pyp.title(r'Lambda billiard. Model: $A = {0}$, $s = {1:.2f}$'.format(a, s), fontsize=fontsize)

def plotAllCover():
    pyp.figure(1)
    pyp.subplot(221)
    plotCoverVsAlpha('Nt9', 90, 0.55)
    pyp.subplot(222)
    plotCoverVsAlpha('Nt8', 80, 0.26)
    pyp.subplot(223)
    plotCoverVsAlpha('Nt7', 70, 0.15)
    pyp.subplot(224)
    plotCoverVsAlpha('Nt5', 50, 0.06)
    pyp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyp.savefig('/home/benjamin/Documents/Temp/lambdaBilliardAvsAlpha.eps', format='eps', dpi=1000)
    
def model(x, a, b):
    return a * b * x / (1 + b * x)