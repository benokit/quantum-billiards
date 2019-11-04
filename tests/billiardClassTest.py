import math as m

import matplotlib.pyplot as plt
import numpy as np

from src import husimiFunctions as hf
from src import spectrumUtilities as su
from src import verginiSaraceno as vs
from src import billiardClass as bc

def circle(N):
    phi = np.linspace(0,2*np.pi,N)
    x = np.cos(phi)
    y = np.sin(phi)
    nx = x
    ny = y
    length = 2*np.pi
    area = np.pi
    return x, y, nx, ny, phi, length, area

x, y, nx, ny, phi, length, area = circle(200)
circleBilliard = bc.billiard(x, y, nx, ny, phi, length, area)

bc.plot_boundary(circleBilliard)
plt.show()

#Vergini Saraceno
k0 = 12
dk = 0.25
N = 100
circleBilliard.scaling_eigenvalues(N, k0, dk)

#Plane Wave decomposition
k_values = np.linspace(1.9, 3, 250)
tensions = [circleBilliard.PWD_tension(N,k) for k in k_values]
plt.semilogy(k_values, tensions)
plt.show()

k0 = 2.4
dk = 0.1
circleBilliard.PWD_eigenvalue(N, k0, dk)

#Probability
plot_probability(circleBilliard, 2.4048256875874907, grid = 400)
plt.show()