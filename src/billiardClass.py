import math as m

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from . import husimiFunctions as hf
from . import spectrumUtilities as su
from . import verginiSaraceno as vs

def midpoints(array):
    """ helper function returns array of half distances between points in array""" 
    return (array[1:] + array[:-1])/2

def integration_weights(boundary_s, L):
    """ Returns weights for discretized integrals on the boundary
    boundary_s are the arc length coordinates of the points an L is the total arc length """
    mdpts = midpoints(boundary_s) 
    right_dist = mdpts - boundary_s[:-1] #half intervals to the right
    right_dist = np.concatenate( [right_dist, [L-boundary_s[-1]] ] ) #append final point
    left_dist = boundary_s[1:] - mdpts #half intervals to the left
    left_dist = np.concatenate([ [boundary_s[0]], left_dist]) # prepend initial point
    return left_dist + right_dist


class billiard:
    """Quantum billiard with no reflection symmetry"""
    # It's a constructor, innit! (read with Cockney accent) #
    def __init__(self, boundary_x, boundary_y, normal_x, normal_y, boundary_s, length, area):
        self.M = len(boundary_x) # number of boundary points
        self.boundary_x = np.array(boundary_x) # x corrdinates of boundary points
        self.boundary_y = np.array(boundary_y) # y coordinates of boundary points
        self.normal_x = np.array(normal_x) # outer normal x coordinates at boundary points
        self.normal_y = np.array(normal_y) # outer normal y coordinates at boundary points
        self.length = length # billiard arc length
        self.area = area # billiard surface area for PWD
        self.boundary_s = boundary_s # arc length coordinates

        rn = (boundary_x * normal_x + boundary_y * normal_y) #scalar product of normal and point vectors
        self.weights = integration_weights(boundary_s, length) #integration weights
        self.PWDweights = self.weights*area/length * rn # A/M *(r*n) weights for PWD needs to be divided by k**2
        self.VSweights = self.weights/rn # L/M * 1/(r*n) weights for scaling method

    def scaling_eigenvalues(self, N, k0, dk):
        """wavenumber eigenvalues of the billiard 
        in the interval [k0-dk, k0+dk] 
        calculated with the Vergini-Saraceno scaling method"""
        F, Fk = vs.ffk_2pi(N, k0, self.VSweights, self.boundary_x, self.boundary_y)
        return vs.eigvals(k0, dk, F, Fk)
    
    def PWD_tension(self, N, k0):
        """Tension of Plane Wave Decomposition at k0. When tension reaches minimum an eigenvalue of k is found"""
        w = self.weights
        wg = self.PWDweights
        x = self.boundary_x
        y = self.boundary_y
        nx = self.normal_x
        ny = self.normal_y
        F, G = vs.fg_2pi(N, k0, w, wg, x, y, nx, ny)
        return vs.eigvalsPWD(k0, F, G)
    
    def PWD_eigenvalue(self, N, k0, dk):
        return optimize.minimize_scalar(lambda x: self.PWD_tension(N,x), bounds=(k0-dk, k0+dk), method='bounded')
    
    def PWD_eigenfunction(self, N, k0, x, y):
        w = self.weights
        wg = self.PWDweights
        bnd_x = self.boundary_x
        bnd_y = self.boundary_y
        nx = self.normal_x
        ny = self.normal_y
        F, G = vs.fg_2pi(N, k0, w, wg, bnd_x, bnd_y, nx, ny)
        vec = vs.eigPWD(k0,F,G)
        return vs.psi_2pi(k0,vec,x,y) 
        

#############################################
#           Plotting functions              #
#############################################


def plot_boundary(billiard):
    x = billiard.boundary_x
    y = billiard.boundary_y
    nx = billiard.normal_x
    ny = billiard.normal_y
    xmin = np.min(x) - 0.15
    xmax = np.max(x) + 0.15
    ymin = np.min(y) - 0.15
    ymax = np.max(y) + 0.15
    plt.plot(x, y, lw = 1.5, color = "k")
    plt.quiver(x,y,nx,ny)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.tight_layout()               

def plot_probability(billiard, k, grid = 400):
    PWDMIN = 100 
    N = max(3 * m.ceil(k / 4), PWDMIN) #number of plane waves
    #grid size
    boundary_x = billiard.boundary_x
    boundary_y = billiard.boundary_y
    xmin = np.min(boundary_x) - 0.05
    xmax = np.max(boundary_x) + 0.05
    ymin = np.min(boundary_y) - 0.05
    ymax = np.max(boundary_y) + 0.05
    
    #coordinates for plot
    q = np.linspace(xmin, xmax, grid+1)
    q2  = np.linspace(ymin, ymax, grid+1)
    Xplot = q
    Yplot = q2
    
    #coordinates for wavefunction
    q = midpoints(q) 
    q2 = midpoints(q2)
    x = np.tile(q, grid)
    y = np.repeat(q2, grid)
    X = np.reshape(x, (grid, grid))
    Y = np.reshape(y, (grid, grid))
    
    #plot billiard boundary
    col="0.5"
    lw=1.5
    ax = plt.gca()#plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
    ax.axis('off')
    ax.set_aspect('equal', 'box')
    ax.plot(boundary_x,boundary_y,col,lw=lw)
    plt.xlabel(r"x")
    plt.ylabel(r"y")

    #calculate probability    
    psi = billiard.PWD_eigenfunction(N, k, X, Y)
    repsi = psi.real
    impsi = psi.imag
    Z = repsi*repsi + impsi*impsi
    Z = np.reshape(Z, (grid,grid))
    vmax = np.max(Z)

    #plot probability
    ax.pcolormesh(Xplot, Yplot, Z, cmap='magma', vmin=0, vmax=vmax)
    plt.tight_layout()