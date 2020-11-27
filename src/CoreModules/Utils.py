import numpy as np

def midpoints(array):
    """ helper function returns array of half distances between points in array""" 
    return (array[1:] + array[:-1])/2

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def reflect_wavefunction(A, sym_x, sym_y):
    """Helper function extends array A from upper right quadrant
    to the whole plane using reflections in accordance with the symmetry class.
    - sym is the symmetry class sym = 0 even_even, sym = 1 even_odd, sym=2 for odd_even and sym=3 for odd_odd 
    Used for poltting the wavefunctions.
    """
    #no symmetry
    if sym_x == None and sym_y == None:
        return A
        
    # only x symmetry
    if sym_x == "odd" and sym_y == None:
        B = -np.flipud(A)
        C = np.concatenate((B, A), axis=0)
        return C
    if sym_x == "even" and sym_y == None:
        B = np.flipud(A)
        C = np.concatenate((B, A), axis=0)
        return C

    # only y symmetry
    if sym_y == "odd" and sym_x == None:
        B = -np.fliplr(A)
        C = np.concatenate((B, A), axis=0)
        return C

    if sym_y == "even" and sym_x == None:
        B = np.fliplr(A)
        C = np.concatenate((B, A), axis=0)
        return C

    # both x and y symmetry
    if sym_x == "odd" and sym_y == "odd":
        B= -np.flipud(A)
        C= np.concatenate((B, A), axis=0)
        D = C[:,:]
        F = -np.fliplr(C)
        E = np.concatenate((F, D), axis=1)
        return E

    if sym_x == "odd" and sym_y == "even":
        B= np.flipud(A)
        C= np.concatenate((B, A), axis=0)
        D = C[:,:]
        F = -np.fliplr(C)
        E = np.concatenate((F, D), axis=1)
        return E

    if sym_x == "even" and sym_y == "odd":
        B= -np.flipud(A)
        C= np.concatenate((B, A), axis=0)
        D = C[:,:]
        F = np.fliplr(C)
        E = np.concatenate((F, D), axis=1)
        return E

    if sym_x == "even" and sym_y == "even":
        B= np.flipud(A)
        C= np.concatenate((B, A), axis=0)
        D = C[:,:]
        F = np.fliplr(C)
        E = np.concatenate((F, D), axis=1)
        return E

def define_plot_area(boundary_x, boundary_y, sym_x, sym_y):
    if sym_x == None and sym_y == None:
        xmin = np.min(boundary_x) - 0.05
        xmax = np.max(boundary_x) + 0.05
        ymin = np.min(boundary_y) - 0.05
        ymax = np.max(boundary_y) + 0.05
    elif sym_x is not None and sym_y == None:
        xmin = np.min(boundary_x) - 0.05
        xmax = np.max(boundary_x) + 0.05
        ymin = 0
        ymax = np.max(boundary_y) + 0.05
    elif sym_y is not None and sym_x == None:
        xmin = 0
        xmax = np.max(boundary_x) + 0.05
        ymin = np.min(boundary_y) - 0.05
        ymax = np.max(boundary_y) + 0.05
    else:
        xmin = 0
        xmax = np.max(boundary_x) + 0.05
        ymin = 0
        ymax = np.max(boundary_y) + 0.05
    return xmin, xmax, ymin, ymax


def reflect_plot_area(x_plot,y_plot, sym_x, sym_y):
    #no symmetry
    if sym_x == None and sym_y == None:
        Xplot = x_plot
        Yplot = y_plot
    elif sym_x is not None and sym_y == None:
        Xplot = np.concatenate((-x_plot[::-1][:-1],x_plot))
        Yplot = y_plot
    elif sym_y is not None and sym_x == None:
        Xplot = x_plot
        Yplot = np.concatenate((-y_plot[::-1][:-1],y_plot))
    else:
        Xplot = np.concatenate((-x_plot[::-1][:-1],x_plot))
        Yplot = np.concatenate((-y_plot[::-1][:-1],y_plot))
    return Xplot, Yplot

def reflect_boundary(bnd_x, bnd_y, sym_x, sym_y):
    boundary_x = bnd_x
    boundary_y = bnd_y
    if sym_x == None and sym_y == None:
        return boundary_x, boundary_y
    elif sym_x is not None and sym_y == None:
        boundary_x = np.concatenate((boundary_x, boundary_x[::-1]))
        boundary_y = np.concatenate((boundary_y, -boundary_y[::-1]))
        return boundary_x, boundary_y
    elif sym_y is not None and sym_x == None:
        boundary_x = np.concatenate((boundary_x, -boundary_x[::-1]))
        boundary_y = np.concatenate((boundary_y, boundary_y[::-1]))
        return boundary_x, boundary_y
    else:
        boundary_x = np.concatenate((boundary_x, -boundary_x[::-1]))
        boundary_y = np.concatenate((boundary_y, boundary_y[::-1]))
        boundary_x = np.concatenate((boundary_x, boundary_x[::-1]))
        boundary_y = np.concatenate((boundary_y, -boundary_y[::-1]))
    return boundary_x, boundary_y