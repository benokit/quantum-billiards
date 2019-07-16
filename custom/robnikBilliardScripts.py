#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 20:49:46 2017

@author: benjamin
"""

from ..src import robnikBilliard as rb
import numpy as np
import pandas as pd
import collections 
from ..src import brodyAnalysis
import pandas as pd
import functools as ft

#SPECT_FOLDER = "/home/benjamin/Research/QuantumChaos/RobnikQuantum/Spectrum/"

#LAMBDAS = [0.25,0.24,0.23,0.22,0.21,0.2,0.19,0.18,0.17,0.16]
    
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
           
def robnikSpectrumAndSpacings(lam, k0, dk, chunk, nchunk):
    spect_filename_root = "robnik_energ_lam={0}_k0={1}_".format(lam, k0)
    # asym
    spect_funk = lambda k: rb.robnikSpecter_asym(lam, k, dk, chunk)
    computeSpectrum(spect_filename_root + "l", spect_funk, k0, nchunk)
    # sym
    spect_funk = lambda k: rb.robnikSpecter_sym(lam, k, dk, chunk)
    computeSpectrum(spect_filename_root + "s", spect_funk, k0, nchunk)

def readSpectInfos(lam, k0, folder = ""):
    spect_filename_root = folder + "robnik_energ_lam={0}_k0={1}_".format(lam, k0)
    SpectInfo = collections.namedtuple('SpectInfo', 'vaweNumbers spacings')
    spectInfos = []
    # asym
    #energies = np.loadtxt(spect_filename_root + "l")
    spect_filename_root = folder + "robnik_energ_lam={0}_k0={1}_".format(lam, k0)
    energies = pd.read_csv(spect_filename_root + "l",sep = '\s+',index_col=False, header = None)
    energies = np.array(energies).transpose()[0]
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(rb.robnikSpectrumUnfold_asym(lam, energies)))
    spectInfos.append(SpectInfo(vaweNumbers, spacings))
    # sym
    #energies = np.loadtxt(spect_filename_root + "s")
    energies = pd.read_csv(spect_filename_root + "s",sep = '\s+',index_col=False, header = None)
    energies = np.array(energies).transpose()[0]
    vaweNumbers = np.sqrt(energies[:-1])
    spacings = np.diff(np.sort(rb.robnikSpectrumUnfold_sym(lam, energies)))
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

def betaVsVaweNumberAnalysis(spectInfos, n, rho = 0, fixedRho = False):
    minK = min([np.min(x.vaweNumbers) for x in spectInfos])
    maxK = max([np.max(x.vaweNumbers) for x in spectInfos])
    ks = np.linspace(minK, maxK, (n + 1))
    ks = ks + (maxK - minK) / (2 * n)
    ks = ks[0:n]
    dk = (ks[1] - ks[0]) / 2
    betas = []

    if fixedRho:
        rhos = []
        for k in ks: 
            spacings = getSpacingsInInterval(spectInfos, k, dk)
            spacings = spacings / np.mean(spacings)
            rho, beta = brodyAnalysis.brodyBRFit(spacings, rho = rho, fixedRho = True)
            betas.append(beta)
            rhos.append(rho)
    else:
        rhos = []
        for k in ks: 
            spacings = getSpacingsInInterval(spectInfos, k, dk)
            spacings = spacings / np.mean(spacings)
            rho, beta = brodyAnalysis.brodyBRFit(spacings, fixedRho = False)
            betas.append(beta)
            rhos.append(rho)

    return ks, np.array(betas), np.array(rhos)



def collectBrodyAnalysis(lambdas, k0, folder="", samples=12, fixedRho = False):
    AnalysisInfo = collections.namedtuple("AnalysisInfo", "lam, ks, betas, rhos")
    analysis = []

    if fixedRho:        
        rhoDatafolder = r"/net/lozej/QuantumBilliards/QuantumRobnikBilliard/LevelRepulssion"
        rhoDataPath = rhoDatafolder + r"/RhoDataRobnik.csv"

        rhoData = pd.read_csv( rhoDataPath, sep = ',',index_col=False)
        lams, rhos0 = np.array(rhoData).transpose()

        for lam in lambdas:
            spectInfo = readSpectInfos(lam, k0, folder)
            rho = 1 - rhos0[lams == lam][0]
            print(lam)
            print(rho)
            #print(spectInfo)
            ks, betas, rhos = betaVsVaweNumberAnalysis(spectInfo, samples, rho=rho, fixedRho=True)
            analysis.append(AnalysisInfo(lam, ks, betas, rhos))

    else:
        for lam in lambdas:
            spectInfo = readSpectInfos(lam, k0, folder)
            #print(spectInfo)
            ks, betas, rhos = betaVsVaweNumberAnalysis(spectInfo, samples)
            analysis.append(AnalysisInfo(lam, ks, betas, rhos))
    return analysis

def convertAnalysisInfosToDataFrame(analysisInfos):
    q = [np.vstack((np.repeat(x.lam,len(x.ks)),x.ks,x.betas,x.rhos)) for x in analysisInfos]
    z = ft.reduce(lambda x, y: np.hstack((x,y)),q[1:],q[0])
    df = pd.DataFrame.from_dict({'lam': z[0],'k': z[1], 'beta': z[2], 'rho': z[3]})
    return df[['lam','k','beta','rho']]
        
    