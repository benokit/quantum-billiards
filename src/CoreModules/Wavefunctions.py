import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
from . import Utils as ut
from . import Solvers as sol
from . import HusimiFunctions as hf
from . import HusimiFunctionsOld as hfold

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
            if not isinstance(delta, list):
                M =  k*delta/(2*np.pi)
            else:
                M = [k*i/(2*np.pi) for i in delta]    
            bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = False,
                                                      midpts = True, normal = True, weights = True)
        if self.solver == "DM":
            ten, vec = sol.decomposition_method(k, self.basis, bnd_pts, L,  eps = self.eps, return_vector = True)
        if self.solver == "PSM":
            int_pts = self.billiard.random_interior_points(Mi)
            ten, vec = sol.particular_solutions_method(k, self.basis, bnd_pts, int_pts, eps = self.eps, return_vector = True)
        return vec

    def scaling_eigenvectors(self, k0, dk, bnd_pts = None, delta = 5, return_ks = True):
        #L = self.billiard.length
        #A = self.billiard.area
        if bnd_pts is None:
            if not isinstance(delta, list):
                M =  k0*delta/(2*np.pi)
            else:
                M = [k0*i/(2*np.pi) for i in delta]    
            bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = False,
                                                      midpts = True, normal = True, weights = True)
        ks, ten, X = sol.scaling_method(k0, dk, self.basis, bnd_pts, return_vector=True)
        
        if return_ks:
            return ks, ten, X
        else:
            return X

        
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
            
            if not isinstance(delta, list):
                M =  k*delta/(2*np.pi)
            else:
                M = [k*i/(2*np.pi) for i in delta]
      
            Mi = int(np.max(M)*L/2)
            #print(Mi)   
            bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = False,  midpts = True, normal = True, weights = True)

            vec = self.eigenvector(k, bnd_pts, Mi=Mi)
        else:
            #print([len(vec)])
            self.basis.set_basis_size([len(vec)])
                                                    
        B = self.basis.evaluate_basis(k, x, y)
        psi = np.transpose(B).dot(vec)
        return psi

    def grad(self,k,x,y, delta = 5, vec = None):
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
            
            if not isinstance(delta, list):
                M =  k*delta/(2*np.pi)
            else:
                M = [k*i/(2*np.pi) for i in delta]
      
            Mi = int(np.max(M)*L/2)
            #print(Mi)   
            bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = False,  midpts = True, normal = True, weights = True)

            vec = self.eigenvector(k, bnd_pts, Mi=Mi)
        else:
            self.basis.set_basis_size([len(vec)])
                                                    
        G_x, G_y = self.basis.evaluate_gradient(k, x, y)
        gx = np.transpose(G_x).dot(vec)
        gy = np.transpose(G_y).dot(vec)
        return gx, gy
            
    def u(self,k, delta = 5, vec = None, midpts = True):

        if not isinstance(delta, list):
            M =  k*delta/(2*np.pi)
        else:
            M = [k*i/(2*np.pi) for i in delta]    
        
        bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = True,  midpts = midpts, normal = True, weights = True)
        bnd_x, bnd_y, nx, ny, bnd_s, weights = bnd_pts.x, bnd_pts.y, bnd_pts.nx, bnd_pts.ny, bnd_pts.s, bnd_pts.ds

        if vec is None:
            L = self.billiard.length
            #A = self.billiard.area
            n_funct = len(self.basis.basis_functions)
            Mi = int(np.max(M)*L/2)   

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
        else:
            self.basis.set_basis_size([len(vec)])
    
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
            if not isinstance(delta, list):
                M =  k*delta/(2*np.pi)
            else:
                M = [k*i/(2*np.pi) for i in delta]    
            bnd_pts = self.billiard.evaluate_boundary(M, evaluate_virtual = False,  midpts = True, normal = True, weights = True)
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
            else:
                self.basis.set_basis_size([len(vec)])

            #construct normalization matrix
            U = self.basis.evaluate_u(k, bnd_x, bnd_y, nx, ny)  #Transposed boundary function
            u = np.transpose(U).dot(vec)
            rn = (bnd_x * nx + bnd_y * ny)
            u_weights = weights  * rn /(2*k**2)
            scalar_product = np.sum(u_weights * u**2)
            return np.sqrt(scalar_product)

        if method == "montecarlo":
            int_pts = self.billiard.random_interior_points(Mi, bnd_pts_dens = 100)
            x,y = int_pts.x, int_pts.y
            Mint = len(x)
            plt.plot(x,y, ".")
            #print(Mint/Mi)
            self.billiard.plot_boundary()
            psi = self.psi(k, x, y, delta=delta)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            scalar_product = np.mean(Z)*A
            return np.sqrt(scalar_product)

        if method == "grid":
            int_pts = self.billiard.interior_points(Mi, bnd_pts_dens = 100)
            x,y = int_pts.x, int_pts.y
            Mint = len(x)
            plt.plot(x,y, ".")
            #print(Mint/Mi)
            self.billiard.plot_boundary()
            psi = self.psi(k, x, y, delta=delta)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            scalar_product = np.mean(Z)*A
            return np.sqrt(scalar_product)

    #not yet final version!!!!!!!!!!!!!!!!
    def Husimi(self, k, qs, ps, delta = 5, vec = None, ver = "new"):
        #L = self.billiard.length 
        s, u, ds = self.u(k, vec = vec, delta = delta)
        #periodize u
        s, u, ds = self.continue_u(s, u, ds)
        #print("u size = %s" %len(u))
        #print("u size = %s" %len(u))
        if ver == "new":
            H = hf.husimiOnGrid(k, s, ds, u, qs, ps)
        if ver == "old":
            H = hfold.husimiOnGrid(k, s, ds, u, qs, ps)
        return H

    def plot_amplitude_histogram(self, k, vec = None, g = 10, bins = 20, delta = 5, kind = "real", dtype = np.float32):
        #grid size
        L = self.billiard.length
        sym_x = self.sym_x
        sym_y = self.sym_y
        bnd_pts = self.billiard.evaluate_boundary(2*np.pi*10/L, 
                                                evaluate_virtual = True, evaluate_sym=True, normal=False, midpts = False)
        boundary_x, boundary_y = bnd_pts.x, bnd_pts.y
        xmin, xmax, ymin, ymax = ut.define_plot_area(boundary_x, boundary_y, sym_x, sym_y)
        #coordinates for plot
        grid_x = int(g*np.abs(xmax-xmin)*k/(2*np.pi))
        grid_y = int(g*np.abs(ymax-ymin)*k/(2*np.pi))

        #coordinates for plot
        q = np.linspace(xmin, xmax, grid_x+1, dtype = dtype)
        q2  = np.linspace(ymin, ymax, grid_y+1, dtype = dtype)
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
        
        #calculate probability    
        psi = np.zeros(grid_x*grid_y, dtype = np.float32)
        psi[inside] = self.psi(k, xx[inside], yy[inside], delta=delta, vec = vec)
        
        repsi = psi[inside].real
        impsi = psi[inside].imag
        Z = repsi*repsi + impsi*impsi

        if kind == "square":
            h = plt.hist(Z, bins=bins, density=True, histtype= "step", color = 'k')
            #plt.plot((bins[1:] + bins[:-1])/2, h) 
        if kind == "imag":
            h = plt.hist(impsi, bins=bins, density=True, histtype= "step", color = 'k')
            #plt.plot((bins[1:] + bins[:-1])/2, h) 
        if kind == "real":
            h = plt.hist(repsi, bins=bins, density=True, histtype= "step", color = 'k')
            #plt.plot((bins[1:] + bins[:-1])/2, h) 
        #vmax = np.max(Z)
        return h



    def plot_probability(self, k, vec = None, g = 5, cmap='binary', plot_exterior = False, delta = 5, plot_full=False, axis=False, dtype = np.float32, col_max = 1):
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
        grid_x = max(int(g*np.abs(xmax-xmin)*k/(2*np.pi)), 200)
        grid_y = max(int(g*np.abs(ymax-ymin)*k/(2*np.pi)), 200)
        #coordinates for plot
        q = np.linspace(xmin, xmax, grid_x+1, dtype = dtype)
        q2  = np.linspace(ymin, ymax, grid_y+1, dtype = dtype)
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
            psi = self.psi(k, xx, yy, delta=delta, vec = vec).astype(dtype)
            repsi = psi.real
            impsi = psi.imag
            Z = repsi*repsi + impsi*impsi
            vmax = np.max(Z[inside])
            Z = np.reshape(Z, (grid_y,grid_x))
            if plot_full:
                psi = np.reshape(psi, (grid_y,grid_x))
                psi = ut.reflect_wavefunction(psi, sym_x, sym_y)
                repsi = psi.real
                impsi = psi.imag
                Z = repsi*repsi + impsi*impsi

        else:
            #calculate probability    
            psi = np.zeros(grid_x*grid_y, dtype = dtype)
            psi[inside] = self.psi(k, xx[inside], yy[inside], delta=delta, vec = vec).astype(dtype)
            psi = np.reshape(psi, (grid_y,grid_x))
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
            self.billiard.plot_boundary( M = 2*np.pi*100/L, normal = False, color = "k", origin = False)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
    
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        if not axis:
            ax = plt.gca()#plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
            ax.axis('off')

        #plot probability
        plt.pcolormesh(Xplot, Yplot, Z, cmap=cmap, vmin=0, vmax=vmax*col_max)


    def plot_nodal_lines(self, k, vec = None, g = 5, cmap='binary', plot_exterior = False, delta = 5, plot_full=False, axis=False, dtype = np.float32, col_max = 1):
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
        grid_x = max(int(g*np.abs(xmax-xmin)*k/(2*np.pi)), 200)
        grid_y = max(int(g*np.abs(ymax-ymin)*k/(2*np.pi)), 200)
        #coordinates for plot
        q = np.linspace(xmin, xmax, grid_x+1, dtype = dtype)
        q2  = np.linspace(ymin, ymax, grid_y+1, dtype = dtype)
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
            psi = self.psi(k, xx, yy, delta=delta, vec = vec).astype(dtype)
            repsi = psi.real
            nodal = repsi
            nodal[nodal>0] = 1
            nodal[nodal<0] = -1
            Z = nodal
            Z = np.reshape(Z, (grid_y,grid_x))
            if plot_full:
                psi = np.reshape(psi, (grid_y,grid_x))
                psi = ut.reflect_wavefunction(psi, sym_x, sym_y)
                repsi = psi.real
                nodal = repsi
                nodal[nodal>0] = 1
                nodal[nodal<0] = -1
                Z = nodal

        else:
            #calculate probability    
            psi = np.zeros(grid_x*grid_y, dtype = dtype)
            psi[inside] = self.psi(k, xx[inside], yy[inside], delta=delta, vec = vec).astype(dtype)
            psi = np.reshape(psi, (grid_y,grid_x))
            if plot_full:
                psi = ut.reflect_wavefunction(psi, sym_x, sym_y)
            repsi = psi.real
            nodal = repsi
            nodal[nodal>0] = 1
            nodal[nodal<0] = -1
            Z = nodal
            #vmax = np.max(Z)


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
            self.billiard.plot_boundary( M = 2*np.pi*100/L, normal = False, color = "k", origin = False)
            plt.xlim(xmin,xmax)
            plt.ylim(ymin,ymax)
    
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        if not axis:
            ax = plt.gca()#plt.axes(xlim=(-1-eps-0.05, 1+eps+0.05), ylim=(-1-eps, 1+eps))
            ax.axis('off')

        #plot probability
        plt.pcolormesh(Xplot, Yplot, Z, cmap=cmap, vmin=0, vmax=0.5)

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

    def plot_boundary_function_histogram(self, k, delta = 5, vec = None, midpts=True):
        s, u, ds = self.u(k, delta = delta, vec = vec, midpts=midpts)
        # plots boundary points of the curves as vertical lines
        data = u/k
        print(len(data))
        plt.hist(data, bins=100, density=True, histtype= "step", color = 'k')
        plt.ylabel('$P(u/k)$')
        plt.xlabel('$u/k$')
    
    #not yet final version!!!!!!!!!!!!!!!
    def plot_Husimi_function(self, k , delta = 5, vec = None, q_grid = 400, p_grid = 400, plot_curve_bounds = True, cmap='binary', retHus = False, col_max = 1, ver = "new"):
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
        H = self.Husimi(k, qs, ps, delta = delta, vec = vec, ver=ver)
        vmax = np.max(H)

        ax = plt.gca()
        #plot Husimi function
        ax.pcolormesh(Qplot, Pplot, H, cmap=cmap, vmin=0, vmax=vmax*col_max)

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

        #plt.xlabel(r"$q$")
        #plt.ylabel(r"$p$")
        if retHus:
            return Qplot, Pplot, H