import math as m

import matplotlib.pyplot as plt
import numpy as np

def midpoints(array):
    """ helper function returns array of half distances between points in array""" 
    return (array[1:] + array[:-1])/2


#############################################
#           Plotting functions              #
#############################################
def plot_curve(curve, M = 50):
    x, y, nx, ny, bnd_s = curve.evaluate(M)
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

def plot_boundary(billiard, M = 10):
    L = billiard.length
    x, y, nx, ny, bnd_s = billiard.evaluate_boundary(2*np.pi*M/L)
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

def plot_tension(billiard, kmin, kmax, N = 200, grid = 200):
    k_vals = np.linspace(kmin, kmax, grid)
    tensions = [billiard.PWD_tension(N, k) for k in k_vals]
    plt.semilogy(k_vals,tensions)
    plt.xlabel("k")
    plt.tight_layout()


def plot_probability(billiard, k, grid = 400):
    PWDMIN = 100 
    N = max(3 * m.ceil(k / 4), PWDMIN) #number of plane waves
    #grid size
    L = billiard.length
    boundary_x, boundary_y, nx, ny, bnd_s = billiard.evaluate_boundary(2*np.pi*10/L)
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