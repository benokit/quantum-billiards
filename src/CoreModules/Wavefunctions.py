import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from . import Utils as ut
from . import Solvers as sol
from . import HusimiFunctions as hf

class wavefunctions:

    def __init__(self, billiard, basis, solver = "DM", scale_basis = None, eps = 0.5e-15, sym_x = None, sym_y = None):
        self.billiard = copy.deepcopy(billiard)
        self.basis = copy.deepcopy(basis)
        self.scale_basis = scale_basis
        self.eps = eps
        self.solver = solver
        self.sym_x = sym_x
        self.sym_y = sym_y

    def eigenvector(self, k, bnd_pts = None, delta = 5, Mi = 100):
        L = self.billiard.length
        A = self.billiard.area
        if bnd_pts is None:
            density = delta*k/(2*np.pi)
            bnd_pts = self.billiard.evaluate_boundary(density, evaluate_virtual = False,
                                                      midpts = True, normal = True, weights = True)
        if self.solver == "DM":
            ten, vec = sol.decomposition_method(k, self.basis, bnd_pts, L,  eps = self.eps, return_vector = True)
        if self.solver == "PSM":
            int_pts = self.billiard.random_interior_points(Mi)
            ten, vec = sol.particular_solutions_method(k, self.basis, bnd_pts, int_pts, eps = self.eps, return_vector = True)
        return vec

        
    def psi(self,k,x,y, delta = 5, vec = None):
        L = self.billiard.length
        n_funct = len(self.basis.basis_functions)
        
        if vec is None:
            if self.scale_basis is not None:
                if not isinstance(self.scale_basis, list):
                    b = np.array([self.scale_basis for i in range(n_funct)])
                    #print("is not list")
                    #print(b)
                else:
                    b = self.scale_basis
                    #print(b)
                self.basis.set_basis_size([int(np.ceil(k*L*i/(2*np.pi))) for i in b])
            
            density = delta*k/(2*np.pi)
            Mi = density*L/2
            bnd_pts = self.billiard.evaluate_boundary(density, evaluate_virtual = False,  midpts = True, normal = True, weights = True)

            vec = self.eigenvector(k, bnd_pts, Mi=Mi)
                                                    
        B = self.basis.evaluate_basis(k, x, y)
        psi = np.transpose(B).dot(vec)
        return psi

            
    def u(self,k, delta = 5, vec = None, midpts = True):

        density = delta*k/(2*np.pi)
        bnd_pts = self.billiard.evaluate_boundary(density, evaluate_virtual = True,  midpts = midpts, normal = True, weights = True)
        bnd_x, bnd_y, nx, ny, bnd_s, weights = bnd_pts.x, bnd_pts.y, bnd_pts.nx, bnd_pts.ny, bnd_pts.s, bnd_pts.ds

        if vec is None:
            L = self.billiard.length
            #A = self.billiard.area
            n_funct = len(self.basis.basis_functions)
            Mi = density*L/2

            if self.scale_basis is not None:
                if not isinstance(self.scale_basis, list):
                    b = np.array([self.scale_basis for i in range(n_funct)])
                    #print("is not list")
                    #print(b)
                else:
                    b = self.scale_basis
                    #print(b)
                self.basis.set_basis_size([int(np.ceil(k*L*i/(2*np.pi))) for i in b])
                            
            vec = self.eigenvector(k, bnd_pts, Mi=Mi)
    
        #construct normalization matrix
        U = self.basis.evaluate_u(k, bnd_x, bnd_y, nx, ny)  #Transposed boundary function
        u = np.transpose(U).dot(vec)

        return bnd_s, u, weights

    def continue_u(self, s, u, ds):
        #no symmetry
        L = self.billiard.length
        sym_x = self.sym_x
        sym_y = self.sym_y
        if sym_x == None and sym_y == None:
            s2 = np.concatenate([s-L, s, L+s])
            u2 = np.concatenate([u, u, u])
            ds2 = np.concatenate([ds, ds, ds])
            return s2, u2, ds2
        # only x symmetry
        if sym_x == "odd" and sym_y == None:
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([-u[::-1],u, -u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2
        if sym_x == "even" and sym_y == None:
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([u[::-1],u, u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        # only y symmetry
        if sym_y == "odd" and sym_x == None:
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([-u[::-1],u, -u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        if sym_y == "even" and sym_x == None:
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([u[::-1],u, u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        # both x and y symmetry
        if sym_x == "odd" and sym_y == "odd":
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([-u[::-1],u, -u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        if sym_x == "odd" and sym_y == "even":
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([-u[::-1],u, u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        if sym_x == "even" and sym_y == "odd":
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([u[::-1],u, -u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

        if sym_x == "even" and sym_y == "even":
            s2 = np.concatenate([-s[::-1], s, 2*L -s[::-1]])
            u2 = np.concatenate([u[::-1],u, u[::-1]])
            ds2 = np.concatenate([ds[::-1],ds, ds[::-1]])
            return s2, u2, ds2

    def norm(self, k, method = "boundary", delta = 5, vec = None , Mi = 1e5):
        L = self.billiard.length
        A = self.billiard.area
    
        if method == "boundary":
            density = delta*k/(2*np.pi)
            bnd_pts = self.billiard.evaluate_boundary(density, evaluate_virtual = False,  midpts = True, normal = True, weights = True)
            bnd_x, bnd_y, nx, ny, weights = bnd_pts.x, bnd_pts.y, bnd_pts.nx, bnd_pts.ny, bnd_pts.ds

            L = self.billiard.length
            A = self.billiard.area
            n_funct = len(self.basis.basis_functions)
            if vec is None:
                if self.scale_basis is not None:
                    if not isinstance(self.scale_basis, list):
                        b = np.array([self.scale_basis for i in range(n_funct)])
                        #print("is not list")
                        #print(b)
                    else:
                        b = self.scale_basis
                        #print(b)
                    self.basis.set_basis_size([int(np.ceil(k*L*i/(2*np.pi))) for i in b])
                            
                ten, vec = sol.decomposition_method(k, self.basis, bnd_pts, L, eps = self.eps, return_vector = True)
            
            #construct normalization matrix
            U = self.basis.evaluate_u(k, bnd_x, bnd_y, nx, ny)  #Transposed boundary function
            u = np.transpose(U).dot(vec)
            rn = (bnd_x * nx + bnd_y * ny)
            u_weights = weights/L  * rn /(2*k**2)
            scalar_product = np.sum(u_weights * u**2)
            return np.sqrt(scalar_product)

        if method == "montecarlo":
            int_pts = self.billiard.random_interior_points(Mi, bnd_pts_dens = 20)
            x,y = int_pts.x, int_pts.y
            psi = self.psi(k, x, y, delta=delta)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            scalar_product = np.mean(Z)
            return np.sqrt(scalar_product)*A

    #not yet final version!!!!!!!!!!!!!!!!
    def Husimi(self, k, qs, ps, delta = 5, vec = None):
        #L = self.billiard.length 
        s, u, ds = self.u(k, vec = vec, delta = delta)
        #periodize u
        s, u, ds = self.continue_u(s, u, ds)
        H = hf.husimiOnGrid(k, s, ds, u, qs, ps)
        return H

    


    def plot_probability(self, k, vec = None, grid = 400, cmap='binary', plot_exterior = False, delta = 5, plot_full=False, axis=False):
        """Plots the probability distribution of the wavefunction at wavevector k.
        The wavefunction is computed using the plane wave decomposition method.
        - k is the eigen wavenumber  
        - grid is tne number of grid points in one dimension
        """
        #grid size
        L = self.billiard.length
        sym_x = self.sym_x
        sym_y = self.sym_y
        bnd_pts = self.billiard.evaluate_boundary(2*np.pi*10/L, 
                                                evaluate_virtual = True, evaluate_sym=True, normal=False, midpts = False)
        boundary_x, boundary_y = bnd_pts.x, bnd_pts.y
        xmin, xmax, ymin, ymax = ut.define_plot_area(boundary_x, boundary_y, sym_x, sym_y)
        #coordinates for plot
        q = np.linspace(xmin, xmax, grid+1)
        q2  = np.linspace(ymin, ymax, grid+1)
        Xplot = q
        Yplot = q2
           
        #coordinates for wavefunction
        q = ut.midpoints(q) 
        q2 = ut.midpoints(q2)
        X,Y = np.meshgrid(q,q2)

        # find points inside of polygon        
        polygon = np.array([boundary_x, boundary_y]).T #array of boundary points [x,y] 
        xx = X.ravel()
        yy = Y.ravel()
        points = np.array((xx, yy)).T
        path = mpltPath.Path(polygon) 
        inside = path.contains_points(points) #indices of points inside polygon

        if plot_exterior:
            psi = self.psi(k, xx, yy, delta=delta, vec = vec)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            vmax = np.max(Z[inside])
            Z = np.reshape(Z, (grid,grid))
            if plot_full:
                psi = np.reshape(psi, (grid,grid))
                psi = ut.reflect_wavefunction(psi, sym_x, sym_y)
                repsi = psi.real
                impsi = psi.imag
                Z = repsi*repsi + impsi*impsi

        else:
            #calculate probability    
            psi = np.zeros(grid*grid)
            psi[inside] = self.psi(k, xx[inside], yy[inside], delta=delta, vec = vec)
            psi = np.reshape(psi, (grid,grid))
            if plot_full:
                psi = ut.reflect_wavefunction(psi, sym_x, sym_y)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            vmax = np.max(Z)

        if plot_full:
            Xplot, Yplot = ut.reflect_plot_area(Xplot, Yplot, sym_x, sym_y)
            bnd_pts = self.billiard.evaluate_boundary(2*np.pi*10/L, 
                                                evaluate_virtual = False, evaluate_sym=False, normal=False, midpts = False)
            bnd_x, bnd_y = bnd_pts.x, bnd_pts.y
            bnd_x, bnd_y = ut.reflect_boundary(bnd_x, bnd_y, sym_x, sym_y)
            col="0.5"
            lw=1.5
            ax = plt.gca()
            ax.plot(bnd_x, bnd_y,col,lw=lw)
            ax.set_aspect('equal', 'box')

        #plot billiard boundary
        else:
            self.billiard.plot_boundary( M = 2*np.pi*10/L, normal = False, color = "k", origin = False)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
    
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        if not axis:
            ax = plt.gca()#plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
            ax.axis('off')

        #plot probability
        plt.pcolormesh(Xplot, Yplot, Z, cmap=cmap, vmin=0, vmax=vmax)

    #not yet final version!!!!!!!!!!!!!!!!
    def plot_boundary_function(self, k, delta = 5, vec = None, plot_curve_bounds = True, midpts=False):
        s, u, ds = self.u(k, delta = delta, vec = vec, midpts=midpts)
        # plots boundary points of the curves as vertical lines
        col = "0.75"
        lw = 0.75
        if plot_curve_bounds:
            L = 0
            plt.axvline(x=L, color = col, lw=lw)
            for crv in self.billiard.curves:
                L = L + crv.length
                if crv.virtual:
                    if not crv.symmetry:
                        plt.axvline(x=L, color = col, lw=lw, ls = "--")
                else:
                    plt.axvline(x=L, color = col, lw=lw)

        plt.plot(s,u)
        plt.xlabel(r"$q$")
        plt.ylabel(r"$u$")
    
    #not yet final version!!!!!!!!!!!!!!!
    def plot_Husimi_function(self, k , delta = 5, vec = None, q_grid = 400, p_grid = 400, plot_curve_bounds = True, cmap='binary'):
        #grid size
        L = self.billiard.length
        #coordinates for plot
        qs = np.linspace(0, L, q_grid+1)
        ps  = np.linspace(0, 1, p_grid+1)
        Qplot = qs
        Pplot = ps
        
        #coordinates for Husimi function
        qs = ut.midpoints(qs) 
        ps = ut.midpoints(ps)
        #calculate Husimi function
        H = self.Husimi(k, qs, ps, delta = delta, vec = vec)
        vmax = np.max(H)

        ax = plt.gca()
        #plot Husimi function
        ax.pcolormesh(Qplot, Pplot, H, cmap=cmap, vmin=0, vmax=vmax)

        # plots boundary points of the curves as vertical lines
        col = "0.75"
        lw = 0.75
        if plot_curve_bounds:
            L = 0
            plt.axvline(x=L, color = col, lw=lw)
            for crv in self.billiard.curves:
                L = L + crv.length
                if crv.virtual:
                    if not crv.symmetry:
                        plt.axvline(x=L, color = col, lw=lw, ls = "--")
                else:
                    plt.axvline(x=L, color = col, lw=lw)

        plt.xlabel(r"$q$")
        plt.ylabel(r"$p$")