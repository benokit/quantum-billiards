import numpy as np
import matplotlib.pyplot as plt
#import BasisFunction as bf

class basis:
    """A class used to represent a basis of the Hilbert space.
    
    A basis object is used to define the basis of Helmholtz solutions,
    that span the Hilbert space and are used to solve the billiard eigenvalue problem.
    The basis object is passed as an argument to the spectrum and wavefunction classes.

    Multiple basis functions of diffrent kinds may be used in the same basis.
    For instance both sine and cosine plane waves may be used to construct the basis.
    See the BasisModules folder for examples of basis object construction. 

    Attributes
    ----------
    basis_functions : list of basis_function objects
        A list of basis_function objects that define the basis.
        See the BasisFunction module for more information.
    basis_size : int
        Dimension of the Hilbert space basis.
    
    Methods
    -------

    set_basis_size(ns)
        Sets the number of basis functions of each kind used in the basis.

    evaluate_basis(k, x, y)
        Evaluates basis and returns matrix of values of each basis function at each point.

    evaluate_u(k, x, y, nx, ny)
        Evaluates basis and returns matrix of directional derivatives of each basis function at each point.

    evaluate_df_dk(k, x, y)
        Evaluates basis and returns matrix of k derivatives of each basis function at each point.

    plot_basis_function(i,kind,k)
        Visualisation function. Plots selected basis function. 

    plot_basis(self, k)
        Visualisation function. Plots first 9 basis functions of each kind. 

    add_basis_function(self, basis_fun, min_size = 10)
        Adds basis function to basis.
    """

    
    # It's a constructor, innit! #
    def __init__(self, basis_functions, min_size = 200):
        """
        Parameters
        ----------
        basis_functions : list of basis function objects
            The basis functions used to define the basis of the Hilbert space.
            Multiple basis functions of diffrent kinds may be used in the same basis.
            See the BasisModules folder for examples of basis object construction.
        min_size : int, optional
            Number of each kind of basis function at initialization.
            (default is 10)
        """
        self.min_size = min_size
        self.basis_functions = basis_functions
        self.basis_size = [min_size for i in range (len(basis_functions))]
        
    #def add_basis_function(basis_function)
    
    def set_basis_size(self, ns):
        """
        Sets the number of basis functions of each kind used in the basis.
        
        Use this method to change the basis dimension.

        Parameters
        ----------
        ns : list 
            The new dimensions of the basis. 
            The size of the list of integers must corespond to the size of the list of basis functions.
            This allows for different sizes for different basis functions.   
        """
        bs = self.basis_size
        new = [self.min_size for i in range(len(bs))]
        for i in range(len(bs)):
            if ns[i]> self.min_size:
                new[i] = ns[i]
        self.basis_size = new
        
    def evaluate_basis(self, k, x, y):
        """
        Evaluates basis and returns matrix of values of each basis function at each point.
        
        Parameters
        ----------
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.

        Returns
        -------
        A : numpy array
            Matrix of basis function values. The size of the matrix is (n,m),
            where n is the basis dimension and m is the number of evaluation points.
            The matrix element A[i,j] is the basis function with index i,
            evaluated at point j.
        """
        result = []
        sz = len(self.basis_size)
        for i in range(sz):
            n = self.basis_size[i]
            bf = self.basis_functions[i]
            A =  np.array([bf.f(j,k,x,y,n = n) for j in range(n)])
            #js = [j for j in range(n)]
            #A = np.array(list(map(lambda j: bf.f(j,k,x,y,n = n), js)))
            result.append(A)
        
        return np.concatenate(result)

    def evaluate_gradient(self, k, x, y):
        """
        Evaluates gradient and returns matrix of values of the gradient each basis function at each point.
        
        Parameters
        ----------
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.

        Returns
        -------
        A : numpy array
            Matrix of basis function values. The size of the matrix is (n,m),
            where n is the basis dimension and m is the number of evaluation points.
            The matrix element A[i,j] is the basis function with index i,
            evaluated at point j.
        """
        result = []
        sz = len(self.basis_size)
        for i in range(sz):
            n = self.basis_size[i]
            bf = self.basis_functions[i]
            A =  np.array([bf.grad_f(j,k,x,y,n = n) for j in range(n)])
            #js = [j for j in range(n)]
            #A = np.array(list(map(lambda j: bf.f(j,k,x,y,n = n), js)))
            result.append(A)
        G = np.concatenate(result)
        A, B =np.split(G,2,axis = 1)
        dim, trash, npts = A.shape
        G_x = A.reshape(dim,npts) 
        G_y = B.reshape(dim,npts)
        return G_x, G_y
    
    def evaluate_u(self, k, x, y, nx, ny):
        """
        Evaluates basis and returns matrix of directional derivatives of each basis function at each point.

        Parameters
        ----------
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

        Returns
        -------
        U : numpy array
            Matrix of basis directional derivatives. The size of the matrix is (n,m),
            where n is the basis dimension and m is the number of evaluation points.
            The matrix element U[i,j] is the basis directional derivative with index i,
            evaluated at point j.
        """
        result = []
        sz = len(self.basis_size)
        for i in range(sz):
            n = self.basis_size[i]
            bf = self.basis_functions[i]
            #A =  np.array([bf.u_f(j,k,x,y,nx,ny,n = n) for j in range(n)])
            js = [j for j in range(n)]
            A = np.array(list(map(lambda j: bf.u_f(j,k,x,y,nx,ny,n = n), js)))
            result.append(A)
    
        return np.concatenate(result)

    def evaluate_df_dk(self, k, x, y):
        """
        Evaluates basis and returns matrix of values of each basis function at each point.
        
        Parameters
        ----------
        k : float
            The wavenumber.
        x : float or numpy array
            The x coordinates of the evaluation points.
        y : float or numpy array
            The y coordinates of the evaluation points.

        Returns
        -------
        F : numpy array
            Matrix of k derivatives. The size of the matrix is (n,m),
            where n is the basis dimension and m is the number of evaluation points.
            The matrix element F[i,j] is the k derivative of the basis function with index i, 
            evaluated at point j.
        """
        result = []
        sz = len(self.basis_size)
        for i in range(sz):
            n = self.basis_size[i]
            bf = self.basis_functions[i]
            #A =  np.array([bf.df_dk(j,k,x,y, n = n) for j in range(n)])
            js = [j for j in range(n)]
            A = np.array(list(map(lambda j: bf.df_dk(j,k,x,y, n = n), js)))
            result.append(A)
    
        return np.concatenate(result)
    
    def plot_basis_function(self,i,kind,k, cmap='RdBu', xlim = (-1,1), ylim= (-1,1)):
        """Visualisation function. Plots selected basis function.

        Parameters
        ----------
        i : int
            The index of the basis function.
        kind : int
            The index of of the list of basis functions.
            Selects the type of basis function.        
        k : float
            The wavenumber.
        """
        n = int(self.basis_size[kind])
        #print(n)
        self.basis_functions[kind].plot_fun(i,k,n =n, cmap=cmap, xlim = xlim, ylim= ylim)

    def plot_basis(self, k):
        """Visualisation function. Plots first 9 basis functions of each kind.
        
        Used to check the basis is constructed correctly.

        Parameters
        ----------
        k : float
            The wavenumber.
        """
        dim = len(self.basis_functions)    
        for j in range(dim):
            bf = self.basis_functions[j]
            bs = self.basis_size[j]
            sz = int(np.min([9, bs]))
            
            #fig = plt.figure(figsize=figsize)
            for i in range(sz):
                plt.subplot(3,3,i+1)
                bf.plot_fun(i, k, n = bs)
            plt.tight_layout()
    
    def add_basis_function(self, basis_fun, min_size = 300):
        """Adds basis function to basis.
        
        Parameters
        ----------
        basis_fun : basis_function object
        The basis function we wish to add to the basis.
        min_size : int
        The number of new basis functions.
        """
        self.basis_functions.append(basis_fun)
        self.basis_size.append(min_size)

def combine_basis(basis1, basis2):
    """Creates a new basis containing basis functions from two basis objects.
    
    Parameters
    ----------
    basis1 : basis object 
        The first basis.
    basis2 : basis object
        The second basis.
    
    Returns
    -------
    bas : basis object.
        New basis object containing basis functions from both imput basis.
    """
    bf1 = basis1.basis_functions
    #print(bf1)
    bs1 = basis1.basis_size

    bf2 = basis2.basis_functions
    #print(bf2)
    bs2 = basis2.basis_size
    
    bf = bf1.copy()
    bs = bs1.copy()
    for i in range(len(bf2)):
        bf.append(bf2[i])
        bs.append(bs2[i])
    #print(bf1)
    bas = basis(bf)
    bas.set_basis_size(bs)
    return bas