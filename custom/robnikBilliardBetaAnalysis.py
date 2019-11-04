#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:14:18 2017

@author: benjamin
"""

import pandas as pa
import matplotlib.pyplot as pyp
from scipy.optimize import curve_fit

alphaDataPathOld = '/home/benjamin/Research/QuantumChaos/RobnikQuantum/report/transportTimesRobnik.csv'

alphaDataPath = '/home/benjamin/Research/QuantumChaos/Python/LambdaBilliardTransportTimes.csv'

betaDataPath = '/home/benjamin/Research/QuantumChaos/Python/betaDataRobnikBilliard.csv'

def plotBetaVsAlpha(ntcol = 'Nt5', percent = 50, markersize=2, fontsize=8):
    dfb = pa.read_csv(betaDataPath)
    dfa = pa.read_csv(alphaDataPath)
    df = dfb.join(dfa.set_index('lam'), on='lam')
    x = 2*df['k']/df[ntcol]
    y = df['beta']
    pyp.plot(x,y,'bo', markersize=markersize)
    ii = x.values.argsort()
    xx = x.values[ii]
    yy = y.values[ii]
    popt, pcov = curve_fit(model, xx, yy, bounds=(0,10))
    pyp.plot(xx, model(xx, *popt), 'r-', label='fit')
    pyp.xlabel(r'$\alpha({0}\%)$'.format(percent))
    pyp.ylabel(r'$\beta$')
    pyp.title(r'Robnik billiard. Model: $A = {0}$, $s = {1:.2f}$'.format(0.96, popt[0]), fontsize=fontsize)
    return popt, pcov

def plotBetaVsK(markersize=2, fontsize=8):
    df = pa.read_csv(betaDataPath)
    grouping = df.groupby('lam')
    for name, group in grouping:
        k = group['k']
        beta = group['beta']
        popt, pcov = curve_fit(model, k, beta, bounds=(0,[10]))
        y = beta
        pyp.plot(k,y,label = r'$lam = {0}, A = {1:.2f}, s = {2:.2f}$'.format(name,0.96,1/popt[0]))
        pyp.plot(k, model(k, *popt), 'b:')
    #pyp.legend()

def plotAll():
    pyp.figure(1)
    pyp.subplot(221)
    plotBetaVsAlpha('Nt9', 90)
    pyp.subplot(222)
    plotBetaVsAlpha('Nt8', 80)
    pyp.subplot(223)
    plotBetaVsAlpha('Nt7', 70)
    pyp.subplot(224)
    plotBetaVsAlpha('Nt6', 60)
    pyp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyp.savefig('/home/benjamin/Documents/Temp/robnikBilliardBvsA.eps', format='eps', dpi=1000)
    
def model(x, b):
    return 0.98 * b * x / (1 + b * x)

def model2(x, a, b):
    return a * b * x / (1 + b * x)