#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:49:46 2017

@author: benjamin
"""

import stadiumBilliard as sb
import numpy as np
import collections 
import brodyAnalysis
import matplotlib.pyplot as pyp
import pandas as pd
import functools as ft

SPECT_FOLDER = "/home/benjamin/Research/QuantumChaos/StadiumQuantum/Spectrum/"

EPSILONS = [0.01,0.0125,0.015,0.0175,0.02,0.0225,0.025,0.0275,0.03,0.0325,0.035,0.0375]#,0.04,0.0425,0.045,0.0475,0.05,0.0525,0.055,0.0575,0.06,0.0625,0.065,0.0675,0.07,0.0725,0.075,0.0775,0.08,0.0825,0.085,0.0875,0.09,0.0925,0.095,0.0975,0.1,0.1025,0.105,0.1075]
    
def appendToFile(filename, x):
    f = open(filename, "ab")
    np.savetxt(f, x)
    f.close()
    
def computeSpectrum(spect_filename, spect_funk, k0, n):
    kk = k0
    for n in range(1, n):
        ks = spect_funk(kk)
        es = ks * ks
        appendToFile(spect_filename, es)
        kk = ks[-1]
           
def stadiumSpectrumAndSpacings(eps, k0, dk, chunk, nchunk):
    spect_filename_root = "stadium_energ_eps={0}_k0={1}_".format(eps, k0)
    # asym_sym
    spect_funk = lambda k: sb.stadiumSpecter_asym_sym(eps, k, dk, chunk)
    computeSpectrum(spect_filename_root + "ls", spect_funk, k0, nchunk)
    # sym_asym
    spect_funk = lambda k: sb.stadiumSpecter_sym_asym(eps, k, dk, chunk)
    computeSpectrum(spect_filename_root + "sl", spect_funk, k0, nchunk)
    # sym_sym
    spect_funk = lambda k: sb.stadiumSpecter_sym_sym(eps, k, dk, chunk)
    computeSpectrum(spect_filename_root + "ss", spect_funk, k0, nchunk)
    # asym_asym
    spect_funk = lambda k: sb.stadiumSpecter_asym_asym(eps, k, dk, chunk)
    computeSpectrum(spect_filename_root + "ll", spect_funk, k0, nchunk)

def readSpectInfos(eps, k0, folder = ""):
    spect_filename_root = folder + "stadium_energ_eps={0}_k0={1}_".format(eps, k0)
    SpectInfo = collections.namedtuple('SpectInfo', 'vaweNumbers spacings')
    spectInfos = []
    # asym_sym
    energies = np.loadtxt(spect_filename_root + "ls")
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(sb.stadiumSpectrumUnfold_asym_sym(eps, energies)))
    spectInfos.append(SpectInfo(vaweNumbers, spacings))
    # sym_asym
    energies = np.loadtxt(spect_filename_root + "sl")
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(sb.stadiumSpectrumUnfold_sym_asym(eps, energies)))
    spectInfos.append(SpectInfo(vaweNumbers, spacings))
    # sym_sym
    energies = np.loadtxt(spect_filename_root + "ss")
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(sb.stadiumSpectrumUnfold_sym_sym(eps, energies)))
    spectInfos.append(SpectInfo(vaweNumbers, spacings))
    # asym_asym
    energies = np.loadtxt(spect_filename_root + "ll")
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(sb.stadiumSpectrumUnfold_asym_asym(eps, energies)))
    spectInfos.append(SpectInfo(vaweNumbers, spacings))
    #return
    return spectInfos

def getSpacingsInInterval_single(spectInfo, k, dk):
    z = spectInfo.vaweNumbers
    s = spectInfo.spacings
    c_a = z > (k - dk)
    c_b = z < (k + dk)
    c = c_a * c_b
    return s[c]

def getSpacingsInInterval(spectInfos, k, dk):
    spacings = np.array([])
    for spectInfo in spectInfos:
        subs = getSpacingsInInterval_single(spectInfo, k, dk)
        spacings = np.concatenate((spacings,subs))
    return spacings

def betaVsVaweNumberAnalysis(spectInfos, n):
    minK = min([np.min(x.vaweNumbers) for x in spectInfos])
    maxK = max([np.max(x.vaweNumbers) for x in spectInfos])
    ks = np.linspace(minK, maxK, (n + 1))
    ks = ks + (maxK - minK) / (2 * n)
    ks = ks[0:n]
    dk = (ks[1] - ks[0]) / 2
    betas = []
    for k in ks: 
        spacings = getSpacingsInInterval(spectInfos, k, dk)
        beta = brodyAnalysis.brodyFit(spacings)
        betas.append(beta)
    return ks, np.array(betas)

def collectBrodyAnalysis(epsilons, k0, folder="", samples=12):
    AnalysisInfo = collections.namedtuple("AnalysisInfo", "eps, ks, betas")
    analysis = []
    for eps in epsilons:
        spectInfo = readSpectInfos(eps, k0, folder)
        ks, betas = betaVsVaweNumberAnalysis(spectInfo, samples)
        analysis.append(AnalysisInfo(eps, ks, betas))
    return analysis

def convertAnalysisInfosToDataFrame(analysisInfos):
    q = [np.vstack((np.repeat(x.eps,len(x.ks)),x.ks,x.betas)) for x in analysisInfos]
    z = ft.reduce(lambda x, y: np.hstack((x,y)),q[1:],q[0])
    df = pd.DataFrame.from_dict({'eps': z[0],'k': z[1], 'beta': z[2]})
    return df[['eps','k','beta']]
    
def plotAnalysisInfos(analysisInfos, kScalingCorrection):
    for analysis in analysisInfos:
        f = kScalingCorrection(analysis.eps)
        label = "{0}".format(analysis.eps)
        pyp.plot(analysis.ks * f, analysis.betas, label=label)
        
def plotAnalysisInfosWithScalingCorrection(analysisInfos):
    plotAnalysisInfos(analysisInfos,lambda x: x**2.5) 
    pyp.xlabel(r'$k\,\epsilon^{2.5}$')
    pyp.ylabel(r'$\beta$')
    pyp.title(r'$\beta$ as a function of $k$ for various values of $\epsilon$')
    lgd = pyp.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyp.savefig("betaVsK_WithScalingCorrection.eps", format="eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyp.show()

def plotAnalysisInfosWithoutScalingCorrection(analysisInfos):
    plotAnalysisInfos(analysisInfos,lambda x: 1)  
    pyp.xlabel(r'$k$')
    pyp.ylabel(r'$\beta$')
    pyp.title(r'$\beta$ as a function of $k$ for various values of $\epsilon$')
    lgd = pyp.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyp.savefig("betaVsK.eps", format="eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyp.show()
    
def plotAnalysisInfosWithTransportTimeCorrection(analysisInfos):
    q = np.loadtxt('TransportTimesStadium2.txt',delimiter='\t',usecols=(0,5))
    plotAnalysisInfos(analysisInfos,lambda x: np.interp(x,q[:,0],2/q[:,1])) 
    pyp.xlabel(r'$2\,k/N_t(exp)$')
    pyp.ylabel(r'$\beta$')
    pyp.title(r'$\beta$ as a function of $k$ for various values of $\epsilon$')
    lgd = pyp.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pyp.savefig("betaVsK_WithTransportTimeCorrection_exp.eps", format="eps", bbox_extra_artists=(lgd,), bbox_inches='tight')
    pyp.show()
        
    