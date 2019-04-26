import math as m

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from . import husimiFunctions as hf
from . import spectrumUtilities as su
from . import verginiSaraceno as vs
from . import planeWaveDecomposition as pwd

def midpoints(array):
    """ helper function returns array of half distances between points in array""" 
    return (array[1:] + array[:-1])/2

def integration_weights(boundary_s, L):
    """ Returns weights for discretized integrals on the boundary.
    - boundary_s are the arc length coordinates of the points
    - L is the arc length """
    mdpts = midpoints(boundary_s) 
    right_dist = mdpts - boundary_s[:-1] #half intervals to the right
    right_dist = np.concatenate( [right_dist, [L-boundary_s[-1]] ] ) #append final point
    left_dist = boundary_s[1:] - mdpts #half intervals to the left
    left_dist = np.concatenate([ [boundary_s[0]], left_dist]) # prepend initial point
    return left_dist + right_dist

class billiard:
    """Convex quantum billiard with no reflection symmetry.
    - curves is a list (or array like object) of curve objects. The curves should form a closed border of the billiard.
    - area is the surface area of the billiard.
    - point_densities is a list of point density parameters for each curve. The number of points evaluated on each curve is density*k*length/(2pi). Default value is 10.
    """
    # It's a constructor, innit! #
    def __init__(self, curves, area, point_densities = False):
        self.curves = curves # 
        self.area = area # x corrdinates of boundary points
        self.length = np.sum([crv.length for crv in curves])
        if point_densities:
            self.point_densities = point_densities 
        else:
            self.point_densities = [10 for i in range(len(curves))]
    
    def evaluate_boundary(self,k):
        """Helper function that evaluates the whole boundary and concatenates result"""
        bnd_x = np.array([])
        bnd_y = np.array([])
        normal_x = np.array([])
        normal_y = np.array([])
        bnd_s = np.array([])
        L = 0
        for i in range(len(self.curves)):
            alpha = self.point_densities[i]
            crv = self.curves[i]
            M = np.ceil(alpha * k * crv.length / (2* np.pi))
            x, y, nx, ny, s = crv.evaluate(M)
            s = s + L
            L = L + crv.length
            bnd_x = np.concatenate([bnd_x, x])
            bnd_y = np.concatenate([bnd_y, y])
            normal_x = np.concatenate([normal_x, nx])
            normal_y = np.concatenate([normal_y, ny])
            bnd_s = np.concatenate([bnd_s, s])
        return bnd_x, bnd_y, normal_x, normal_y, bnd_s
    
    def scaling_eigenvalues(self, N, k0, dk):
        """Wavenumber eigenvalues of the billiard 
        in the interval [k0-dk, k0+dk] 
        calculated with the Vergini-Saraceno scaling method"""
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k0 + dk)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        VSweights = weights/rn
                    
        F, Fk = vs.ffk_2pi(N, k0, VSweights, bnd_x, bnd_y)
        return vs.eigvals(k0, dk, F, Fk)
    
    def PWD_tension(self, N, k0):
        """Tension of Plane Wave Decomposition at k0. When tension reaches minimum an eigenvalue of k is found"""
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k0)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        PWDweights = weights*self.area/self.length * rn
        F, G = pwd.fg_2pi(N, k0, weights, PWDweights, bnd_x, bnd_y, normal_x, normal_y)
        return pwd.eigvalsPWD(k0, F, G)
    
    def PWD_eigenvalue(self, N, k0, dk):
        """Uses the scipy.optimize.minimize_sclar routine to find minimum of tension in the interval [k0-dk, k0+dk]"""
        return optimize.minimize_scalar(lambda x: self.PWD_tension(N,x), bounds=(k0-dk, k0+dk), method='bounded')

    def PWD_eigenfunction(self, N, k0, x, y):
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k0)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        PWDweights = weights*self.area/self.length * rn
        F, G = pwd.fg_2pi(N, k0, weights, PWDweights, bnd_x, bnd_y, normal_x, normal_y)
        vec = pwd.eigPWD(k0,F,G)
        return vs.psi_2pi(k0,vec,x,y)        