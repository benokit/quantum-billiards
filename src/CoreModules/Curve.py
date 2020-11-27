import math as m
import numpy as np
import matplotlib.pyplot as plt
from . import Utils as ut


class curve:
    """ A curve parametrized on the interval t = [0, 1].
    Curves define the geometry of the billiard as a set of functions. 
    To construct a curve one must provide the following functions:
    - r(t, **params) -> (x,y) returns the x and y coordinates of the curve at t as a tuple (or any array like object).
    - normal(t, **params) -> (nx, ny) returns the normal vector of the curve at t as a tuple. 
    - arc_length(t, **params) -> l returns the arc length of the curve up to t.
    Additional curve parameters may be provided as a dictionary of key-word arguments params. 
    """
    def __init__(self, r, normal, arc_length,  virtual = False, symmetry = False,
                distribution_function = None, weight_function = None, **parameters):
        self.r = lambda t: r(t, **parameters) #lambda is used to fix parameters
        self.normal = lambda t: normal(t, **parameters)
        self.arc_length = lambda t: arc_length(t, **parameters)
        self.params = parameters
        self.virtual = virtual 
        self.symmetry = symmetry
        self.length = self.arc_length(1)
        self.dist = distribution_function
        self.weight_function = weight_function
        
    
    def evaluate(self, M,  midpts = True, normal = True, weights = False):
        """Evaluate all the curve functions on a equidistant grid on the interval t = [0,1).
        - M is the number of grid points
        Returns one-dimensional numpy arrays:
        - x values of x coordinates
        - y values of y coordinates
        - nx values of normal coordinates in the x direction
        - ny values of normal coordinates in the y direction
        - s values of arc lenght coordinates
        """
        x, y, nx, ny, s, ds = [],[],[],[],[],[]

        if midpts:
            t_values = ut.midpoints(np.linspace(0, 1, M+1))
        else:
            t_values = np.linspace(0, 1, M)
        if self.dist is not None:
            t_values = self.dist(t_values)
        
        x, y = np.array(self.r(t_values))
        if normal:
            nx, ny = np.array(self.normal(t_values))
        
        s = np.array(self.arc_length(t_values))
        if weights:
            if midpts:
                mdpts = ut.midpoints(s)
                mdpts = np.concatenate([[self.arc_length(0)], mdpts ,[self.arc_length(1)]])
                right_dist = mdpts[1:] - s #half intervals to the right
                left_dist = s - mdpts[:-1] #half intervals to the left
            else:
                mdpts = ut.midpoints(s)
                right_dist = mdpts - s[:-1] #half intervals to the right
                left_dist = s[1:] - mdpts #half intervals to the left
                right_dist = np.concatenate( [right_dist, [self.arc_length(1)]] ) #append final point
                left_dist = np.concatenate([ [self.arc_length(0)], left_dist]) # prepend initial point 
            ds = left_dist + right_dist 
        return x, y, nx, ny, s, ds
    
    

    def plot_curve(self, M = 50, normal = True, color= "k"):
        """Plots the curve and its normal directions. 
        The density of the normal vectors indicates the density of points
        - M is the number of points ploted
        """
        x, y, nx, ny, bnd_s, ds = self.evaluate(M, midpts = False, normal = normal)
        #xmin = np.min(x) - 0.15
        #xmax = np.max(x) + 0.15
        #ymin = np.min(y) - 0.15
        #ymax = np.max(y) + 0.15
        if self.virtual:
            plt.plot(x, y, lw = 1.0, color = color, ls="--")
        else:
            plt.plot(x, y, lw = 1.5, color = color)
            if normal:
                plt.quiver(x,y,nx,ny)
        #ax = plt.gca()
        #ax.set_aspect('equal', 'box')
        #plt.xlim(xmin,xmax)
        #plt.ylim(ymin,ymax)
        #plt.tight_layout()