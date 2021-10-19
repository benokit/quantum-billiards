import math as m
import numpy as np
import matplotlib.pyplot as plt
from . import Utils as ut


class curve:
    """ A class used to represent the curves that 
    compose the boundary of the billiard table.

    A curve class object contains the functions that define the shape of the curve,
    its normal direction and its arc length. 
    The functions should be parametrised on the interval [0,1].
    Curve objects are passed to the billiard class to construct a billiard table.
    See examples in the Geometry module in the BilliardModules folder.

    Attributes
    ----------
    r : function
        A parametric function that represents the curve
        in cartesian coordinates.
    normal : function
        A parametric function that represents the normal 
        of the curve in cartesian coordinates.
    arc_length : function
        A parametric function that represents the arc lengt 
        (natural parametrisation) of the curve.
    params : dictionary
        A dictionary of keyword parameters passed to the curve.
    virtual : bool
        If True the curve is virtual and represents a curve 
        where the boundary conditions are automatically met
        by the basis functions.  
    symmetry : bool
        If True the curve is represents a symmetry axis. 
        All symmetry axis are represented as virtual curves and 
        virtual is set to True.
    length : float
        The length of the curve.
    dist : function or None
        A function that determines the distribution of evaluation points.
        If None the distribution is uniform.
    weight_function : function or None
        A function that determines the weights for boundary integration.
        If None the weight is uniform. 
    """
    def __init__(self, r, normal, arc_length,  virtual = False, symmetry = False,
                distribution_function = None, weight_function = None, **parameters):
        self.r = lambda t: r(t, **parameters) #lambda is used to fix parameters
        self.normal = lambda t: normal(t, **parameters)
        self.arc_length = lambda t: arc_length(t, **parameters)
        self.params = parameters
        self.virtual = virtual 
        self.symmetry = symmetry
        if self.symmetry:
            self.virtual = True
        self.length = self.arc_length(1)
        self.dist = distribution_function
        self.weight_function = weight_function
        """
        Parameters
        ----------
        r : function
            A parametric function that represents the curve in cartesian coordinates.
            The curve must be parametrised on the interval [0,1].
            The function should have the following signature:
            r(r, **kwargs), where
                t : float or numpy array
                    is the value of the parameter and
                **kwargs
                    are arbitrary keyword arguments, 
                    used to pass parameters of the curve.
            The function should return a numpy array [x, y], where x and y are numpy arrays 
            of x and y coordinates of the same size as t.
        normal : function
            A parametric function that represents the curve in cartesian coordinates.
            The function signature should be the sam as that of r (see above).
            The function should return a numpy array [nx, ny], where nx and ny are numpy arrays 
            of the x and y components of the nomal vector of the same size as t.
        arc_length : function
            A parametric function that represents the arc lengt (natural parametrisation)
            of the curve.
            The function signature should be the sam as that of r (see above)
            The function should return a numpy array of the same size as t.
        virtual : bool
            If True the curve is virtual and represents a curve where the boundary conditions
            are automatically met by the basis functions.
            (default is False)  
        symmetry : bool
            If True the curve is represents a symmetry axis. 
            All symmetry axis are represented as virtual curves, virtual = True.
            (default is False)
        dist : function or None
            A function that determines the distribution of evaluation points.
            If None the distribution is uniform.
            (default is None)
        weight_function : function or None
            A function that determines the weights for boundary integration.
            If None the weight is uniform.
            (default is None)
        **parameters : dictionary
            A dictionary of keyword arguments to pass to the functions.
        """
        
    
    def evaluate(self, M,  midpts = True, normal = True, weights = False):
        """Evaluates a set number of points on the curve.
            The points are taken on the parameter interval [0,1] 
            and are distributed according to the dist attribute.
        
        Parameters
        ----------
        M : int
            The number of evaluation points.
        midpts : bool
            If True the endpoints at t=0 and t=1 are not included.
            (default is True)
        normal : bool
            If True the normal vector is also evaluated.
            (default is True)
        wieghts : bool
            If True the integration weights according to the 
            weight_function are computed.
            (default is False)
        
        Returns
        -------
        x : numpy array
            The x coordiantes of the points on the boundary.
        y : numpy array
            The y coordiantes of the points on the boundary.
        nx : numpy array
            The x components of the normal vectors on the boundary.
        ny : numpy array
            The y components of the normal vectors on the boundary.
        s : numpy array
            The arc lengths corresponding to the points on the boundary.
        ds : numpy array
            The integration weights.
        
        If any of the above are not computed an empty array is returned.    
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
        """Visualisation function. Plots the curve and optionally the normal vectors.
        
        Parameters
        ----------
        M : int
            The number of evaluation points.
            (default is 50)
        normal : bool
            If True the normal vectors are ploted.
            (default is True)
        color : string
            Set the color of the plot, using matplotlib color options.
        """
        x, y, nx, ny, bnd_s, ds = self.evaluate(M, midpts = False, normal = normal)
        #xmin = np.min(x) - 0.15
        #xmax = np.max(x) + 0.15
        #ymin = np.min(y) - 0.15
        #ymax = np.max(y) + 0.15
        if self.virtual:
            plt.plot(x, y, color = color, ls="--")
        else:
            plt.plot(x, y, color = color)
            if normal:
                plt.quiver(x,y,nx,ny)
        #ax = plt.gca()
        #ax.set_aspect('equal', 'box')
        #plt.xlim(xmin,xmax)
        #plt.ylim(ymin,ymax)
        #plt.tight_layout()