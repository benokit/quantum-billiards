import math as m
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpltPath
from . import Utils as ut


class points:
    def __init__(self, x=None, y=None, nx=None, ny=None, s=None, ds=None):
        self.x = x 
        self.y = y
        self.nx = nx
        self.ny = ny
        self.s = s
        self.ds = ds

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
        self.length = 0
               
        for crv in curves:
            if not crv.symmetry:
                self.length = self.length + crv.length
        
    
    def evaluate_boundary(self, density, evaluate_virtual = False, evaluate_sym = False,  midpts = True, normal = True, weights = False):
        """Helper function that evaluates the whole boundary and concatenates result"""
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

    def random_interior_points(self, M, bnd_pts_dens = 20):
        L = self.length
        res = self.evaluate_boundary(2*np.pi*bnd_pts_dens/L)
        boundary_x = res.x
        boundary_y = res.y
        #weights = ut.integration_weights(bnd_s, billiard.length)
        xmin = np.min(boundary_x) 
        xmax = np.max(boundary_x) 
        ymin = np.min(boundary_y) 
        ymax = np.max(boundary_y) 
        polygon = np.array([boundary_x, boundary_y]).T #array of boundary points [x,y] 
        n_int_pts = int((xmax-xmin)*(ymax-ymin)/self.area * M)
        xx = np.random.uniform(xmin,xmax, n_int_pts)
        yy = np.random.uniform(ymin,ymax, n_int_pts)
        pts = np.array((xx, yy)).T
        path = mpltPath.Path(polygon) 
        inside = path.contains_points(pts) #finds points inside polygon 
        return points(xx[inside], yy[inside])
    
    def interior_points(self, M, bnd_pts_dens = 20):
        L = self.length
        res = self.evaluate_boundary(2*np.pi*bnd_pts_dens/L)
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
    
    def plot_boundary(self, M = 10, normal = True, color = "k", origin = True):
        """Plots the boundary of the billiard and its normal directions. 
        The relative density of the normal vectors indicates the density of points
        - M is the number of points ploted per curve
        """
        L = self.length
        bnd_pts = self.evaluate_boundary(2*np.pi*M/L, evaluate_virtual = False,  
                                                        midpts = True, normal = False, weights = False)
        x = bnd_pts.x
        y = bnd_pts.y
        xmin = np.min(x) - 0.15
        xmax = np.max(x) + 0.15
        ymin = np.min(y) - 0.15
        ymax = np.max(y) + 0.15
        for i in range(len(self.curves)):
            crv = self.curves[i]
            crv.plot_curve(M = int(2*np.pi*M/(L/crv.arc_length(1))), normal = normal, color = color )
       
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        if origin:
            plt.plot([0],[0], "kx")
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)

    