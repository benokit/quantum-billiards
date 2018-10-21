#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 19:08:05 2017

@author: benjamin
"""

import stadiumBilliard as sb
import spectrumUtilities as su
import husimiFunctions as hf
import matplotlib.pyplot as pyp
import numpy as np
import math as m
import glob

def saveToFile(filename, x):
    f = open(filename, "ab")
    np.savetxt(f, x)
    f.close()

def stadiumHusimiSequence(eps, k0, dk, nn):
    fileBase = "stadium_husimi_eps={0}_k0={1}".format(eps, k0)
    n = 0
    k = k0
    spect, S = sb.stadiumEig_sym_sym(eps, k, dk)
    for i in range(0, spect.size):
        n = n + 1
        _,_,H = sb.stadiumHusimi_sym_sym(eps,spect[i],S[:,i])
        saveToFile("{0}_{1}".format(fileBase, n),H)
    while (n < nn):
        k = k + dk
        rspec, rS = sb.stadiumEig_sym_sym(eps, k, dk)
        q = su.getIndexOfFirstInRightWhichIsNotInLeft(spect, rspec)
        for i in range(q, rspec.size):
            n = n + 1
            _,_,H = sb.stadiumHusimi_sym_sym(eps,rspec[i],rS[:,i])
            saveToFile("{0}_{1}".format(fileBase, n),H)
        spect = su.glueSpectra(spect, rspec)
        
def animateHusimi(eps, k0, directory):
    fileBase = directory + "stadium_husimi_eps={0}_k0={1}".format(eps, k0)
    files = glob.glob(fileBase + "*")
    for file in files:
        H = np.loadtxt(file)
        hf.plotHusimi(H, 0, 0.5*m.pi + eps, 0, 1)
        pyp.pause(5)
        
def averageEntropy(eps, k0, directory):
    fileBase = directory + "stadium_husimi_eps={0}_k0={1}".format(eps, k0)
    files = glob.glob(fileBase + "*")
    def funk(file):
        H = np.loadtxt(file)
        return hf.entropy(H)
    es = np.array(list(map(funk,files)))
    return np.mean(es)

def entropyLocalizationMeasure(eps, k0, directory, fileToAppend):
    fileBase = directory + "stadium_husimi_eps={0}_k0={1}".format(eps, k0)
    files = glob.glob(fileBase + "*")
    indx = 0
    for file in files:
        indx = indx + 1
        H = np.loadtxt(file)
        e = hf.entropy(H)
        c = hf.entropyCover(e,160000)
        with open(fileToAppend, 'a') as file:
            file.writelines("{0};{1};{2};{3}\n".format(eps,k0,indx,c))
            
def correlations(eps, k0, directory, fileToAppend):
    fileBase = directory + "stadium_husimi_eps={0}_k0={1}".format(eps, k0)
    files = glob.glob(fileBase + "*")
    n = len(files)
    for i in range(0,n):
        H1 = np.loadtxt(files[i])
        for j in range(i+1,n):
            H2 = np.loadtxt(files[j])
            c = hf.correlation(H1,H2)
            with open(fileToAppend, 'a') as file:
                file.writelines("{0};{1};{2};{3};{4}\n".format(eps,k0,i,j,c))
    
    
        
        
    
        
        
        
        
        