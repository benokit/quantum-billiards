import numpy as np
import math as m

def unfoldWail(A, S, energies):
    return 0.25 * (A * energies - S * np.sqrt(energies)) / m.pi

def computeSpectrum(k0, dk, n, spek_fun):
    k = k0
    spect = spek_fun(k)
    while (spect.size < n):
        k = k + dk      
        s = spek_fun(k)
        spect = glueSpectra(spect, s)
    spect = spect[spect > k0]
    spect = spect[spect < k]
    return spect

def getIndexOfFirstInRightWhichIsNotInLeft(leftSpectrum, rightSpectrum):
    if (leftSpectrum.size == 0):
        return 0
    nLL = leftSpectrum.size
    nRR = rightSpectrum.size
    nR = np.sum(rightSpectrum <= leftSpectrum[-1]) # number of overlaping levels
    if (nRR == 0):
        return -1
    if (nR >= nLL):
        return 0
    if (nR == 0 or nRR == 1):
        return 0
    if (nR == 1):
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR + 1
        w2 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w1, w2])
        na = m.ceil(n / 2)
        nb = n - na
        return nb
    if (nR == nRR):
        n = nR - 1
        w0 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w0, w1]) - 1
        na = m.ceil(n / 2)
        nb = n - na
        return nb
    else:
        n = nR - 1
        w0 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n  
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR + 1
        w2 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w0, w1, w2]) - 1
        na = m.ceil(n / 2)
        nb = n - na
        return nb

def glueSpectra(leftSpectrum, rightSpectrum):
    if (leftSpectrum.size == 0):
        return rightSpectrum
    nLL = leftSpectrum.size
    nRR = rightSpectrum.size
    nR = np.sum(rightSpectrum <= leftSpectrum[-1]) # number of overlaping levels
    if (nRR == 0):
        return leftSpectrum
    if (nR >= nLL):
        return rightSpectrum
    if (nR == 0 or nRR == 1):
        return np.concatenate((leftSpectrum, rightSpectrum))
    if (nR == 1):
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR + 1
        w2 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w1, w2])
        na = m.ceil(n / 2)
        nb = n - na
        return np.concatenate((leftSpectrum[0:-na], rightSpectrum[nb:nRR]))
    if (nR == nRR):
        n = nR - 1
        w0 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w0, w1]) - 1
        na = m.ceil(n / 2)
        nb = n - na
        return np.concatenate((leftSpectrum[0:-na], rightSpectrum[nb:nRR]))
    else:
        n = nR - 1
        w0 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n  
        n = nR
        w1 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n
        n = nR + 1
        w2 = np.sum(np.abs(leftSpectrum[-n:nLL]-rightSpectrum[0:n])) / n 
        n = nR + np.argmin([w0, w1, w2]) - 1
        na = m.ceil(n / 2)
        nb = n - na
        return np.concatenate((leftSpectrum[0:-na], rightSpectrum[nb:nRR]))
    


