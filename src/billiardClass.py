import math as m

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from . import husimiFunctions as hf
from . import spectrumUtilities as su
from . import verginiSaraceno as vs
from . import wavefunctions as wf
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
        """Computes the wavenumber eigenvalues of the billiard 
        in the interval [k0-dk, k0+dk] 
        using the Vergini-Saraceno scaling method"""
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k0 + dk)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        VSweights = weights/rn
                    
        F, Fk = vs.ffk_2pi(N, k0, VSweights, bnd_x, bnd_y)
        return vs.eigvals(k0, dk, F, Fk)
    
    def PWD_tension(self, N, k):
        """Tension of Plane Wave Decomposition at wavenumber k. When tension reaches minimum an eigenvalue of k is found"""
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        PWDweights = weights*self.area/self.length * rn
        F, G = pwd.fg_2pi(N, k, weights, PWDweights, bnd_x, bnd_y, normal_x, normal_y)
        return pwd.eigvalsPWD(k, F, G)
    
    def PWD_eigenvalue(self, N, k0, dk):
        """Uses the scipy.optimize.minimize_sclar routine to find minimum of tension in the interval [k0-dk, k0+dk]"""
        return optimize.minimize_scalar(lambda x: self.PWD_tension(N,x), bounds=(k0-dk, k0+dk), method='bounded')

    def PWD_eigenfunction(self, N, k, x, y):
        """Wavefunction at wavenumber k constructed using the plane wave decomposition method
            -  N is the number of plane waves
            -  k is the eigen wavenumber 
            -  x and y are 1d arrays of evaluation points
        """
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        PWDweights = weights*self.area/self.length * rn
        F, G = pwd.fg_2pi(N, k, weights, PWDweights, bnd_x, bnd_y, normal_x, normal_y)
        vec = pwd.eigPWD(k,F,G)
        return wf.psi_2pi(k,vec,x,y)

    def boundary_function(self, N, k, delta = 5):
        """Normal derivative of the wavefunction on the boundary at wavenumber k.
           -  N is the number of plane waves
           -  k is the eigen wavenumber 
           -  delta is the scaling factor for the density of points with k
        """
        #boundary
        bnd_x, bnd_y, normal_x, normal_y, bnd_s = self.evaluate_boundary(k*delta)
        weights = integration_weights(bnd_s, self.length)
        rn = (bnd_x * normal_x + bnd_y * normal_y)
        PWDweights = weights*self.area/self.length * rn
        F, G = pwd.fg_2pi(N, k, weights, PWDweights, bnd_x, bnd_y, normal_x, normal_y)
        vec = pwd.eigPWD(k,F,G)
        #gradient
        dpsi_x, dpsi_y = wf.grad_psi_2pi(k, vec, bnd_x, bnd_y,)
        u = dpsi_x * normal_x + dpsi_y * normal_y
        return bnd_s, u

    def Husimi_function(self, N, k, qs, ps, delta = 2):
        """
        Poincare-Husimi function at wavenumber k evaluated on the grid of points given by qs and ps in the quantum phase space.
        - N is the number of plane waves
        - qs is a 1d array of points on the billiard boundary 
        - ps is a 1d array of points in the cannonical momentum
        """
        L = self.length
        s, u = self.boundary_function(N, k, delta = delta)
        ds = integration_weights(s, L)
        #periodize u
        s = np.concatenate([s-L, s, L+s])
        u = np.concatenate([u, u, u])
        ds = np.concatenate([ds, ds, ds]) 
        H = hf.husimiOnGrid(k, s, ds, u, qs, ps)
        return H

    def plot_boundary(self, M = 10):
        """Plots the boundary of the billiard and its normal directions. 
        The relative density of the normal vectors indicates the density of points
        - M is the number of points ploted per curve
        """
        L = self.length
        x, y, nx, ny, bnd_s = self.evaluate_boundary(2*np.pi*M/L)
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

    def plot_tension(self, kmin, kmax, N = 200, grid = 200):
        """Plots the tension as a function of the wavevector k in the interval [kmin, kmax].
        The tension is computed using the plane wave decomposition method.
        - N is the number of plane waves
        - grid is tne number of grid points
        """
        k_vals = np.linspace(kmin, kmax, grid)
        tensions = [self.PWD_tension(N, k) for k in k_vals]
        plt.semilogy(k_vals,tensions)
        plt.xlabel("k")
        plt.tight_layout()


    def plot_probability(self, k, grid = 400, scale = False):
        """Plots the probability distribution of the wavefunction at wavevector k.
        The wavefunction is computed using the plane wave decomposition method.
        - k is the eigen wavenumber  
        - grid is tne number of grid points in one dimension
        """
        PWDMIN = 100 
        N = max(3 * m.ceil(k / 4), PWDMIN) #number of plane waves
        #grid size
        L = self.length
        boundary_x, boundary_y, nx, ny, bnd_s = self.evaluate_boundary(2*np.pi*10/L)
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
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        #calculate probability    
        psi = self.PWD_eigenfunction(N, k, X, Y)
        repsi = psi.real
        impsi = psi.imag
        Z = repsi*repsi + impsi*impsi
        Z = np.reshape(Z, (grid,grid))
        vmax = np.max(Z)
        if scale:
            vmax = vmax * scale

        #plot probability
        ax.pcolormesh(Xplot, Yplot, Z, cmap='magma', vmin=0, vmax=vmax)
        plt.tight_layout()

    def plot_boundary_function(self, k , delta = 5, plot_curve_bounds = True):
        PWDMIN = 100 
        N = max(3 * m.ceil(k / 4), PWDMIN) #number of plane waves
        s, u = self.boundary_function(N, k, delta = delta)
        
        # plots boundary points of the curves as vertical lines
        col = "0.75"
        lw = 0.75
        if plot_curve_bounds:
            L = 0
            plt.axvline(x=L, color = col, lw=lw)
            for crv in self.curves:
                L = L + crv.length
                plt.axvline(x=L, color = col, lw=lw)

        plt.plot(s,u)
        plt.xlabel(r"$q$")
        plt.ylabel(r"$u$")

    def plot_Husimi_function(self, k , delta = 2, q_grid = 400, p_grid = 400, plot_curve_bounds = True):
        PWDMIN = 100 
        N = max(3 * m.ceil(k / 4), PWDMIN) #number of plane waves
        #grid size
        L = self.length
        #coordinates for plot
        qs = np.linspace(0, L, q_grid+1)
        ps  = np.linspace(0, 1, p_grid+1)
        Qplot = qs
        Pplot = ps
        
        #coordinates for Husimi function
        qs = midpoints(qs) 
        ps = midpoints(ps)
        #calculate Husimi function
        H = self.Husimi_function(N, k, qs, ps, delta = delta)
        vmax = np.max(H)

        ax = plt.gca()
        #plot Husimi function
        ax.pcolormesh(Qplot, Pplot, H, cmap='magma', vmin=0, vmax=vmax)

        # plots boundary points of the curves as vertical lines
        col = "0.75"
        lw = 0.75
        if plot_curve_bounds:
            L = 0
            plt.axvline(x=L, color = col, lw=lw)
            for crv in self.curves:
                L = L + crv.length
                plt.axvline(x=L, color = col, lw=lw)

        plt.xlabel(r"$q$")
        plt.ylabel(r"$p$")
        plt.tight_layout()