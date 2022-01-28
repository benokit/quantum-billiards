from libc.math cimport sin, cos, fabs, sqrt, exp
cimport cython
from cython.parallel cimport prange

import numpy as np
cdef double PI = np.pi

cdef int find_idx(double[:] s, double value) nogil:
    cdef int i
    for i in range(len(s)):
        if s[i] > value:
            return i
    return -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double husimiAtPoint(double k, double[:] s, double[:] ds, double[:] u, double q, double p) nogil:
    cdef double width = 4 / sqrt(k)
    cdef double si, ui, dsi, w, cr, ci 
    cdef double hr = 0
    cdef double hi = 0
    cdef int i, start, stop
    #cdef double[:] ss 
    #for i in range(s.shape[0]):
    #    ss[i] = s[i] - q

    start = find_idx(s, -width + q)
    stop = find_idx(s, width + q )
    
    for i in range(start, stop):
        si = s[i] - q
        dsi = ds[i]
        ui = u[i]
        w =   exp(-0.5 * k * si * si) * dsi
        cr = w * cos(k * p * si)
        ci = w * sin(k * p * si)
        hr = hr + cr * ui
        hi = hi + ci * ui
    
    return (hr * hr + hi * hi)/(2*PI*sqrt(k*PI))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void husimiOnGrid_cy(double k, double[:] s, double[:] ds, double[:] u, double[:] qs, double[:] ps, double[:,:] out) nogil:
    """Evaluates the Poicare-Husimi function on the grid given by the arrays qs and ps
       - qs is a 1d array of points on the billiard boundary 
       - ps is a 1d array of points in the cannonical momentum
       - k is the wavenumber of the eigen state
       - s is an array of points on the boundary
       - ds is an array of boundary integration weights
       - u is an array of boundary function values
    """
    cdef int i, j
    for i in prange(ps.shape[0]):
        for j in range(qs.shape[0]):
            out[i][j] = husimiAtPoint(k, s, ds, u, qs[j], ps[i]) 


def husimiOnGrid_py(k, s, ds, u, qs, ps):
    dtyp = u.dtype
    sh = (len(ps),len(qs))
    H = np.zeros(sh, dtype=dtyp)
    husimiOnGrid_cy(k, s, ds, u, qs, ps, H)
    return H / np.sum(H)