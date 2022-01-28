import math as m
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpltPath
from . import Utils as ut


class points:
    """A class used to represent points on the boundary.
    
    Attributes
    ----------
    x : numpy array
        The x coordiantes of the points on the boundary.
    y : numpy array
        The y coordiantes of the points on the boundary.
    nx : numpy array
        The x components of the normal vectors on the boundary.
    ny : numpy array
        The y components of the normal vectors on the boundary.
    s : numpy array
        The arc lengths corresponding to the points 
        on the boundary.
    ds : numpy array
        The integration weights.

    All attributes default to None if not specified.
    """
    def __init__(self, x=None, y=None, nx=None, ny=None, s=None, ds=None):

        self.x = x 
        self.y = y
        self.nx = nx
        self.ny = ny
        self.s = s
        self.ds = ds

class billiard:
    """A class used to represent a billiard table.
    
    A billiard object is a collection of curves that enclose
    a region in the plane.  
    The billiard object is passed as an argument to the spectrum and wavefunction classes.

    Attributes
    ----------
    curves : list of curve objects
        The curves that form the boundary of the billiard table.
        The curves should be listed in order, so that they enclose a
        region in the plane in the positive (counterclockwise) direction. 
        Each curve should begin where the previous curve ends.
        Use virtual virtual curves if the boundary conditions are automatically satisfied
        by the basis on part of the boundary. (see the curve class)
        Use symmetry = True to define a reflection symmetry axis.
    area : float
        The area of the billiard table.
    length : float
        The length of the billiard boundary. 

    Methods
    -------
    evaluate_boundary(density, evaluate_virtual = False, evaluate_sym = False,
                      midpts = True, normal = True, weights = False)
        Evaluates points on the boundary with given point density.

    random_interior_points(self, M, bnd_pts_dens = 20)
        Evaluates a number of random points inside the billiard table.

    interior_points(self, M, bnd_pts_dens = 20)
        Evaluates a number of points inside the billiard table on a grid.

    plot_boundary(self, M = 10, normal = True, color = "k", origin = True)
        Visualisation function. Plots the billiard table boundary.
    """
    # It's a constructor, innit! #
    def __init__(self, curves, area):
        """
        Parameters
        ----------
        curves : list of curve objects
            The curves that form the boundary of the billiard table.
            The curves should be listed in order, so that they enclose a
            region in the plane in the positive (counterclockwise) direction. 
            Each curve should begin where the previous curve ends.
            Use virtual virtual curves if the boundary conditions are 
            automatically satisfied by the basis on part of the boundary.
            (see the curve class)
            Use symmetry = True to define a reflection symmetry axis.
        area : float
            The area of the billiard table.
        """

        self.curves = curves # 
        self.area = area # x corrdinates of boundary points
        self.length = 0
               
        for crv in curves:
            if not crv.symmetry:
                self.length = self.length + crv.length
        
    
    def evaluate_boundary(self, density, evaluate_virtual = False, evaluate_sym = False,  midpts = True, normal = True, weights = False):
        """Evaluates points on the boundary with given point density.
        
        Parameters
        ----------
        density : float
            The linear density of points that is desired.
        evaluate_virtual : bool
            If True the points are evaluated also on the virtual curves.
            (default is False)
        evaluate_sym : bool
            If True the points are evaluated also on the symmetry curves.
            (default is False)
        midpts : bool
            If True the endpoints of each curve are excluded.
            (default is True)
        normal : bool
            If True the normal vector is also evaluated.
            (default is True)
        wieghts : bool
            If True the integration weights according to the 
            weight_function of each curve are computed.
            (default is False)
        Returns
        -------
        pts : points object
            A points object that represents the boundary points.

        """
        bnd_x = np.array([])
        bnd_y = np.array([])
        normal_x = np.array([])
        normal_y = np.array([])
        bnd_s = np.array([])
        bnd_ds = np.array([])
        L = 0
        if not isinstance(density, list):
            dens = [density for i in range(len(self.curves))]
        else:
            dens = density

        if evaluate_virtual:
            for i in range(len(self.curves)):
                alpha = dens[i]
                crv = self.curves[i]
                if not evaluate_sym :
                    if not crv.symmetry:
                        M = int(np.ceil(alpha * crv.length ))
                        x, y, nx, ny, s, ds = crv.evaluate(M, midpts = midpts, normal = normal, weights = weights)
                        s = s + L
                        L = L + crv.length
                        bnd_x = np.concatenate([bnd_x, x])
                        bnd_y = np.concatenate([bnd_y, y])
                        normal_x = np.concatenate([normal_x, nx])
                        normal_y = np.concatenate([normal_y, ny])
                        bnd_s = np.concatenate([bnd_s, s])
                        bnd_ds = np.concatenate([bnd_ds, ds])
                else:
                    M = int(np.ceil(alpha * crv.length ))
                    x, y, nx, ny, s, ds = crv.evaluate(M, midpts = midpts, normal = normal, weights = weights)
                    s = s + L
                    L = L + crv.length
                    bnd_x = np.concatenate([bnd_x, x])
                    bnd_y = np.concatenate([bnd_y, y])
                    normal_x = np.concatenate([normal_x, nx])
                    normal_y = np.concatenate([normal_y, ny])
                    bnd_s = np.concatenate([bnd_s, s])
                    bnd_ds = np.concatenate([bnd_ds, ds])
        else:
            for i in range(len(self.curves)):
                alpha = dens[i]
                crv = self.curves[i]
                if not crv.virtual:
                    M = int(np.ceil(alpha * crv.length ))
                    x, y, nx, ny, s, ds = crv.evaluate(M, midpts = midpts, normal = normal, weights = weights)
                    s = s + L
                    L = L + crv.length
                    bnd_x = np.concatenate([bnd_x, x])
                    bnd_y = np.concatenate([bnd_y, y])
                    normal_x = np.concatenate([normal_x, nx])
                    normal_y = np.concatenate([normal_y, ny])
                    bnd_s = np.concatenate([bnd_s, s])
                    bnd_ds = np.concatenate([bnd_ds, ds])
        
        return points(bnd_x, bnd_y, normal_x, normal_y, bnd_s, bnd_ds)

    def xyn_sample(self, s):
        crv_bounds = [0]
        bound = 0
        for crv in self.curves:
            bound = bound + crv.length
            crv_bounds.append(bound)

        bnd_x = np.array([])
        bnd_y = np.array([])
        normal_x = np.array([])
        normal_y = np.array([])
        for i in range(len(self.curves)):
            if i ==0:
                idx = np.all([s>=crv_bounds[i], s<= crv_bounds[i+1]], axis = 0)
            else:
                idx = np.all([s>crv_bounds[i], s<= crv_bounds[i+1]], axis = 0)
            s_interval = s[idx]
            crv = self.curves[i]
            t_values = (s_interval-crv_bounds[i])/crv.length
            x, y = np.array(crv.r(t_values))
            nx, ny = np.array(crv.normal(t_values))
            bnd_x = np.concatenate([bnd_x, x])
            bnd_y = np.concatenate([bnd_y, y])
            normal_x = np.concatenate([normal_x, nx])
            normal_y = np.concatenate([normal_y, ny])
        return bnd_x, bnd_y, normal_x, normal_y
        

    def random_interior_points(self, M, bnd_pts_dens = 20):
        """Evaluates a number of random points inside the billiard table.
        
        The boundary of the billiard is represented as a polygon
        in order to determine if points are in the interior.
        
        Parameters
        ----------
        M : int
            Number of points that is desired.
        bnd_pts_dens : float
            Density of the points on the boundary used to define the polygon.
            (default is 20)
        
        Returns
        -------
        pts : points object
            A points object that represents the interior points.
        """
        L = self.length
        res = self.evaluate_boundary(2*np.pi*bnd_pts_dens/L, evaluate_virtual=True, evaluate_sym=True)
        boundary_x = res.x
        boundary_y = res.y
        #weights = ut.integration_weights(bnd_s, billiard.length)
        xmin = np.min(boundary_x) 
        xmax = np.max(boundary_x) 
        ymin = np.min(boundary_y) 
        ymax = np.max(boundary_y) 
        polygon = np.array([boundary_x, boundary_y]).T #array of boundary points [x,y] 
        n_int_pts = int(np.sqrt((xmax-xmin)*(ymax-ymin)/self.area * M))
        xx = np.random.uniform(xmin,xmax, n_int_pts)
        yy = np.random.uniform(ymin,ymax, n_int_pts)
        pts = np.array((xx, yy)).T
        path = mpltPath.Path(polygon) 
        inside = path.contains_points(pts) #finds points inside polygon 
        return points(xx[inside], yy[inside])
    
    def interior_points(self, M, bnd_pts_dens = 20):
        """Evaluates a number of points inside the billiard table on a grid.

        The boundary of the billiard is represented as a polygon
        in order to determine if points are in the interior.
        
        Parameters
        ----------
        M : int
            Number of points that is desired.
        bnd_pts_dens : float
            Density of the points on the boundary used to define the polygon.
            (default is 20)
        
        Returns
        -------
        pts : points object
            A points object that represents the interior points.
        """
        L = self.length
        res = self.evaluate_boundary(2*np.pi*bnd_pts_dens/L, evaluate_virtual=True, evaluate_sym=True)
        boundary_x = res.x
        boundary_y = res.y
        #weights = ut.integration_weights(bnd_s, billiard.length)
        xmin = np.min(boundary_x) 
        xmax = np.max(boundary_x) 
        ymin = np.min(boundary_y) 
        ymax = np.max(boundary_y) 
        polygon = np.array([boundary_x, boundary_y]).T #array of boundary points [x,y] 
        n_int_pts = int(np.sqrt((xmax-xmin)*(ymax-ymin)/self.area * M))
        x = np.linspace(xmin,xmax, n_int_pts+1)
        y = np.linspace(ymin,ymax, n_int_pts+1)
        x = ut.midpoints(x) 
        y = ut.midpoints(y)
        X,Y = np.meshgrid(x,y)
        X = X.ravel()
        Y = Y.ravel()
        pts = np.array((X, Y)).T
        path = mpltPath.Path(polygon) 
        inside = path.contains_points(pts) #finds points inside polygon 
        return points(X[inside], Y[inside])
    
    def plot_boundary(self, M = 10, normal = True, color = "k", origin = True, axis = True, lw = 1.0, limits = True):
        """Visualisation function. Plots the billiard table boundary.

        Parameters
        ----------
        M : int
            Number of boundary points.
            (default = 10)
        normal : bool
            If True the normal vectors are ploted.
            (default is True)
        color : string
            Set the color of the plot, using matplotlib color options.
        origin : bool
            If True plots location of origin as an x.
            (default is True)

        """
        L = self.length
        bnd_pts = self.evaluate_boundary(2*np.pi*M/L, evaluate_virtual = True,  
                                                        midpts = True, normal = False, weights = False)
        x = bnd_pts.x
        y = bnd_pts.y
        xmin = np.min(x) - 0.15
        xmax = np.max(x) + 0.15
        ymin = np.min(y) - 0.15
        ymax = np.max(y) + 0.15
        for i in range(len(self.curves)):
            crv = self.curves[i]
            crv.plot_curve(M = int(2*np.pi*M/(L/crv.arc_length(1))), normal = normal, color = color, lw = lw )
       
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if origin:
            plt.plot([0],[0], "kx")
        
        if limits:
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
        
        if not axis:
            ax = plt.gca()#plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
            ax.axis('off')

    