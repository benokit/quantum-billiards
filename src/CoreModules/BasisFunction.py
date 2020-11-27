import matplotlib.pyplot as plt
import numpy as np
from . import Utils as ut

class basis_function:
    def __init__(self, function,  static_params, par_fun = None, gradient = None, k_derivative = None ):
        self.function = function # basis function
        self.gradient = gradient #gradient function
        self.k_derivative = k_derivative
        self.static_params = static_params #dictionary of keyword parameters
        self.par_fun = par_fun
        
            
    def f(self, i, k, x, y, n = False):
        if self.par_fun  is not None:
            scaled_params = self.par_fun(i, n)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.function(i, k, x, y, **par)
    
    def grad_f(self, i, k, x, y, n = False):
        if self.par_fun  is not None:
            scaled_params = self.par_fun(i, n)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.gradient(i, k, x, y, **par)
    
    def u_f(self, i, k, x, y, nx, ny, n = False):
        grd_x, grd_y = self.grad_f( i, k, x, y, n = n)
        return nx*grd_x + ny*grd_y
    
    def df_dk(self, i, k, x, y, n = False):
        if self.par_fun is not None:
            scaled_params = self.par_fun(i, n)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.k_derivative(i, k, x, y, **par)
    
    def plot_fun(self, i, k, n = False, grid = 400, cmap='RdBu', xlim = (-1,1), ylim= (-1,1)):
        xmin = xlim[0]
        xmax = xlim[1]
        ymin = ylim[0]
        ymax = ylim[1]

        #coordinates for plot
        q = np.linspace(xmin, xmax, grid+1)
        q2  = np.linspace(ymin, ymax, grid+1)
        Xplot = q
        Yplot = q2

        #coordinates for function
        q = ut.midpoints(q) 
        q2 = ut.midpoints(q2)
        X,Y = np.meshgrid(q,q2)

        #calculate probability    
        Z = self.f(i, k, X, Y, n = n)
        Z = np.reshape(Z, (grid,grid))
        #vmax = np.max(Z)
     
        ax = plt.gca() #plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
        #ax.axis('off')
        ax.set_aspect('equal', 'box')
        #ax.plot(boundary_x,boundary_y,col,lw=lw)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        #subplot(1,2,1)
        ax = plt.gca() #plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
        #ax.axis('off')
        ax.set_aspect('equal', 'box')
        #ax.plot(boundary_x,boundary_y,col,lw=lw)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        #plot probability
        ax.pcolormesh(Xplot, Yplot, Z, cmap=cmap) #vmin=-vmax, vmax=vmax)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.plot([0],[0], "kx")
        #plt.plot(wedge_x,wedge_y, "k")