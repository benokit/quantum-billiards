#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:14:18 2017

@author: benjamin
"""

import pandas as pa
import matplotlib.pyplot as pyp
from scipy.optimize import curve_fit

alphaAngularDis = '/home/benjamin/Research/QuantumChaos/Python/TransportTimesStadiumAngular.csv'

alphaAngularCon = '/home/benjamin/Research/QuantumChaos/Python/TransportTimesStadiumAngularCont.csv'

betaDataPath = '/home/benjamin/Research/QuantumChaos/Python/betaDataStadium.csv'

def plotBetaVsAlpha(ntcol = 'Nt5', percent = 50, markersize=2, fontsize=8):
    dfb = pa.read_csv(betaDataPath)
    dfa = pa.read_csv(alphaAngularDis)
    df = dfb.join(dfa.set_index('eps'), on='eps')
    x = 2*df['k']/df[ntcol]
    y = df['beta']
    pyp.plot(x,y,'bo', markersize=markersize)
    ii = x.values.argsort()
    xx = x.values[ii]
    yy = y.values[ii]
    popt, pcov = curve_fit(model, xx, yy, bounds=(0,[1,10]))
    pyp.plot(xx, model(xx, *popt), 'r-', label='fit')
    pyp.xlabel(r'$\alpha({0}\%)$'.format(percent))
    pyp.ylabel(r'$\beta$')
    pyp.title(r'Stadium billiard. Model: $A = {0}$, $s = {1:.2f}$'.format(popt[0], popt[1]), fontsize=fontsize)
    return popt, pcov

def plotAll():
    pyp.figure(1)
    pyp.subplot(221)
    plotBetaVsAlpha('Nt9', 90)
    pyp.subplot(222)
    plotBetaVsAlpha('Nt8', 80)
    pyp.subplot(223)
    plotBetaVsAlpha('Nt7', 70)
    pyp.subplot(224)
    plotBetaVsAlpha('Nt5', 50)
    pyp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    pyp.savefig('/home/benjamin/Documents/Temp/stadiumBilliardBvsA.eps', format='eps', dpi=1000)
    
def model(x, a, b):
    return a * b * x / (1 + b * x)