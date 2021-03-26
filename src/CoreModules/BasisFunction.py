import matplotlib.pyplot as plt
import numpy as np
from . import Utils as ut
#######################################################################
class basis_function:
    """A class used to represent basis functions.

    A basis_function object is used to represent a basis function that is a solution
    of the Helmholtz equation. basis_function objects are passed to the Basis class
    in order to costruct a basis for the solution of the billiard eigenvalue problem.
    See the modules in the BasisModules folder for examples.

    Attributes
    ----------
    fun : fun
        The basis function in cartesian coordinates.
    static_params : dictionary
        The paramaters of the basis function that are independent of the basis size and index.
        They are passed to the function as keyword arguments.
    par_fun : function
        A function that returns the parameters that depend on the basis size and index. 
    gradient : function
        The gradient of the basis function in cartesian coordinates.
    k_derivative : function
        The derivative of the basis function with respect to the wavenumber k.

    Methods
    -------
    f(i, k, x, y, n = None)
        Evaluates basis function with index i and wavenumber k at points x, y.

    grad_f(i, k, x, y, n = None)
        Evaluates gradient of basis function with index i and wavenumber k
        at points x, y.

    df_dk i, k, x, y, n = None)
        Evaluates dervative of basis function with regard to k,
        with index i and wavenumber k at points x, y.

    u_f(i, k, x, y, nx, ny, n = None)
        Evaluates directional dervative of basis function,
        with index i and wavenumber k at points x, y,
        with regard to the vector (nx, ny).

    plot_fun(self, i, k, n = None, grid = 400, cmap='RdBu',
            xlim = (-1,1), ylim= (-1,1))
        Visualisation function. Plots the basis function with index i and wavenumber k.
    """

    def __init__(self, fun, static_params, par_fun = None, gradient = None, k_derivative = None ):
        """
        Parameters
        ----------
        fun : function
            The basis function in cartesian coordinates.
            The function should have the following signature:
            fun(i, k, x, y, **kwargs), where
                i : int 
                    is the index of the basis function,
                k : float
                    is the wavenumber,
                x : float or numpy array 
                    is the x coordinates of the evaluation points,
                y : float or numpy array 
                    is the y coordinates of the evaluation points,
                **kwargs
                    are arbitrary keyword arguments, 
                    used to pass parameters to the basis function.
            The function should return a numpy array of values.
        static_params : dictionary
            The paramaters of the basis function that are independent of the basis size and index.
            The dictionary of parameters is passed to the basis function as keyword arguments.
            If no parameters are needed use an empty dictionary {}. 
            Parameters that depend on the index or basis size are handled separately (see below).
        par_fun : function or None
            A function that returns the parameters that depend on the basis size and index.
            (default is None) 
            The function should have the following signature:
            par_fun(i, n), where
                i : int 
                    is the index of the basis function and
                n : int 
                    is the basis size.
            The function should return a dictionary of keyword argumets
            that may be passed to the basis function.             
        gradient : function
            The gradient of the basis function in cartesian coordinates.
            (default is None)
            The function signature should be the same as that of fun.
            The function should return a tuple of numpy arrays df_dx, df_dy, where
                df_dx : numpy array
                    is the derivative with respect to x and
                df_dy : numpy array
                    is the derivative with respect to y.     
        k_derivative : function
            The derivative of the basis function with respect to the wavenumber k.
            (default is None)
            The function signature should be the same as that of fun.
            The function should return a numpy array of derivative values.
        """
        
        self.fun = fun # basis function
        self.gradient = gradient #gradient function
        self.k_derivative = k_derivative
        self.static_params = static_params #dictionary of keyword parameters
        self.par_fun = par_fun
        
            
    def f(self, i, k, x, y, n = None, **kwargs):
        """Evaluates basis function with index i and wavenumber k at point (x, y).
            
        Parameters
        ----------
        i : int
            The index of the basis function.
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.
        n : int, optional
            Size of the basis. 
            Only needed when functional parameters depend on basis size.
            (default is None)
        
        Returns
        -------
        res : numpy array
            Array of basis function values, with the same shape as x and y.
        """

        if self.par_fun  is not None:
            scaled_params = self.par_fun(i, n, **kwargs)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.fun(i, k, x, y, **par)
    
    def grad_f(self, i, k, x, y, n = None, **kwargs):
        """Evaluates gradient of the basis function,
        with index i and wavenumber k at point (x, y).
            
        Parameters
        ----------
        i : int
            The index of the basis function.
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.
        n : int, optional
            Size of the basis. 
            Only needed when functional parameters depend on basis size.
            (default is None)
        
        Returns
        -------
        df_dx, df_dy : tuple of numpy arrays
            Tuple of numpy arrays, with the same shape as x and y.
            df_dx is the derivative with respect to x and
            df_dy is the derivative with respect to y.
        """

        if self.par_fun  is not None:
            scaled_params = self.par_fun(i, n, **kwargs)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.gradient(i, k, x, y, **par)
    
    def u_f(self, i, k, x, y, nx, ny, n = None, **kwargs):
        """Evaluates directional dervative of basis function,
        with index i and wavenumber k at point (x, y),
        with regard to the vector (nx, ny).
        Used to evaluate the billiard boundary functions.
            
        Parameters
        ----------
        i : int
            The index of the basis function.
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.
        nx : float or numpy array
            The x coordinates of the vector.
        ny : float or numpy array
            The y coordinates of the vector.    
        n : int, optional
            Size of the basis. 
            Only needed when functional parameters depend on basis size.
            (default is None)
        
        Returns
        -------
        res : numpy array
            Array of directional derivative values, with the same shape as x and y.
        """
        grd_x, grd_y = self.grad_f( i, k, x, y, n = n)
        return nx*grd_x + ny*grd_y
    
    def df_dk(self, i, k, x, y, n = None, **kwargs):
        """Evaluates dervative of basis function with regard to k,
        with index i and wavenumber k at point (x, y).
        
        Parameters
        ----------
        i : int
            The index of the basis function.
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.
        n : int, optional
            Size of the basis. 
            Only needed when functional parameters depend on basis size.
            (default is None)
            
        Returns
        -------
        res : numpy array
            Array of k derivative values, with the same shape as x and y.
        """
        if self.par_fun is not None:
            scaled_params = self.par_fun(i, n, **kwargs)
            par = {**self.static_params, **scaled_params}
        else: 
            par = self.static_params
        #print(par)
        return self.k_derivative(i, k, x, y, **par)
    
    def plot_fun(self, i, k, n = None,  grid = 400, cmap='RdBu', xlim = (-1,1), ylim= (-1,1),**kwargs):
        """Visualisation function. Plots the basis function with index i and wavenumber k.
        Uses the matplotlib pcolormap function. 
        
        Parameters
        ----------
        i : int
            The index of the basis function.
        k : float
            The wavenumber.
        n : int, optional
            Size of the basis. 
            Only needed when functional parameters depend on basis size.
            (default is None)
        grid : int, optional
            Number of evaluation grid points in one dimension.
            (default is 400)
        cmap : string
            Matplotlib colormap used in the plot.
            (default is 'RdBu')
        xlim : tuple
            Size of plot area in the x direction.
            (default is (-1,1))
        ylim : tuple
            Size of plot area in the y direction.
            (default is (-1,1))
        """
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
        Z = self.f(i, k, X, Y, n = n, **kwargs)
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