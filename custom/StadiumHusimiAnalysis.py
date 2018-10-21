#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:50:29 2018

@author: benjamin
"""

import pandas
#import seaborn
import numpy
import matplotlib.pyplot as pyp
from scipy.optimize import curve_fit

hfolder = '/home/benjamin/Research/QuantumChaos/StadiumQuantum/HusimiAnalysis/'

def plotAvsC():
    
    a = pandas.read_csv(hfolder + 'entropyLocalizationMeasures.dat', delimiter = ";")
    b = a[['eps','k','e']]
    
    aa = b.groupby(['eps','k']).mean().reset_index()
    
    a = pandas.read_csv(hfolder + 'husimi_correl_eps=0.07.dat', delimiter = ";")
    b = a[['eps','k','c']]
    
    q = b.groupby(['eps','k']).mean().reset_index()
    
    a = pandas.read_csv(hfolder + 'husimi_correl_eps=0.08.dat', delimiter = ";")
    b = a[['eps','k','c']]

    q = pandas.concat([q,b.groupby(['eps','k']).mean().reset_index()])
    
    a = pandas.read_csv(hfolder + 'husimi_correl_eps=0.09.dat', delimiter = ";")
    b = a[['eps','k','c']]

    q = pandas.concat([q,b.groupby(['eps','k']).mean().reset_index()])
    
    a = pandas.read_csv(hfolder + 'husimi_correl_eps=0.1.dat', delimiter = ";")
    b = a[['eps','k','c']]

    q = pandas.concat([q,b.groupby(['eps','k']).mean().reset_index()])
    
    bb = q
    
    zz = pandas.merge(aa,bb, on = ['eps','k'])
    
    #seaborn.regplot(zz['e'],zz['c'])
    
betaDataPath = '/home/benjamin/Research/QuantumChaos/Python/betaDataStadium.csv'

def plotBetaVsA():
    a = pandas.read_csv(hfolder + 'entropyLocalizationMeasures.dat', delimiter = ";")
    b = a[['eps','k','e']]
    aa = b.groupby(['eps','k']).mean().reset_index()
    bb = pandas.read_csv(betaDataPath)
    q = numpy.unique(aa['k'])
    bb['k'] = [q[numpy.argmin(numpy.abs(q-k))] for k in bb['k']]
    zz = aa.join(bb.set_index(['eps','k']), on = ['eps','k'])
    #seaborn.regplot(zz['e'],zz['beta'],color='blue')
    xx = zz['e']
    yy = zz['beta']
    pyp.plot(zz['e'],zz['beta'],'bo',markersize = 3)
    popt, pcov = curve_fit(linModel, xx, yy, bounds=([-2,-2],[2,2]))
    pyp.plot(xx, linModel(xx, *popt), 'r-', label='fit')
    pyp.xlabel('A')
    pyp.ylabel(r'$\beta$')
    
def linModel(x, a, b):
    return a * x + b
    
alphaAngularDis = '/home/benjamin/Research/QuantumChaos/Python/TransportTimesStadiumAngular.csv'
    
def plotAVsAlpha(ntcol = 'Nt5', percent = 50, markersize=2, fontsize=8):
    a = pandas.read_csv(hfolder + 'entropyLocalizationMeasures.dat', delimiter = ";")
    b = a[['eps','k','e']]
    dfb = b.groupby(['eps','k']).mean().reset_index()
    dfa = pandas.read_csv(alphaAngularDis)
    df = dfb.join(dfa.set_index('eps'), on='eps')
    x = 2*df['k']/df[ntcol]
    y = df['e']
    pyp.plot(x,y,'bo', markersize=markersize)
    ii = x.values.argsort()
    xx = x.values[ii]
    yy = y.values[ii]
    popt, pcov = curve_fit(model, xx, yy, bounds=(0,10))
    pyp.plot(xx, model(xx, *popt), 'r-', label='fit')
    pyp.xlabel(r'$\alpha({0}\%)$'.format(percent))
    pyp.ylabel(r'$A$')
    #pyp.title(r'Stadium billiard. Model: $A = {0}$, $s = {1:.2f}$'.format(popt[0], popt[1]), fontsize=fontsize)
    return popt, pcov

def plotAll():
    pyp.figure(1)
    pyp.subplot(221)
    plotAVsAlpha('Nt9', 90)
    pyp.subplot(222)
    plotAVsAlpha('Nt8', 80)
    pyp.subplot(223)
    plotAVsAlpha('Nt7', 70)
    pyp.subplot(224)
    plotAVsAlpha('Nt5', 50)
    pyp.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #pyp.savefig('/home/benjamin/Documents/Temp/stadiumBilliardBvsA.eps', format='eps', dpi=1000)
    
def model(x, b):
    return 0.58 * b * x / (1 + b * x)