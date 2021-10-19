from libc.math cimport sin, cos, fabs
cimport cython
from cython.parallel cimport prange

import numpy as np
cdef double PI = np.pi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_sff_parallel(double[:] E, double[:] tim,
                               double[:] weights, double[:] out) nogil:
    cdef int i, j
    cdef double w
    cdef double sff_real
    cdef double sff_imag
 
    for i in prange(tim.shape[0]):
        sff_imag = 0.0
        sff_real = 0.0
        for j in range(E.shape[0]):
            sff_real = sff_real + cos(E[j] * tim[i]* 2*PI) * weights[j]
            sff_imag = sff_imag + sin(E[j] * tim[i]* 2*PI) * weights[j]
        out[i] = sff_real**2 + sff_imag**2

def compute_sff(E, tim , weights):
    out = np.empty_like(tim, dtype='double')
    compute_sff_parallel( E, tim, weights, out)
    return tim, out

def sigma(L, E):
    cdef double Ave1 = 0.0
    cdef double Ave2 = 0.0
    cdef double Ave3 = 0.0
    cdef double Ave4 = 0.0
    cdef int j = 0
    cdef int k = 0
    cdef double x = E[0]
    cdef double s, d1, d2
    cdef int cn
    cdef int max_idx = -(int(L)+100)
    print(max_idx)
    #print(x)
    while x < E[max_idx]:
        #print(k)
        while E[k] < x+L:
            k = k+1
        #continue
        d1 = E[j] - x
        d2 = E[k] - (x+L)
        cn = k - j
        if (d1 < d2):
            x = E[j]
            s = d1
            j = j + 1
        else:
            x = E[k] - L
            s = d2
            k = k + 1
        Ave1 = Ave1 + s*cn
        Ave2 = Ave2 + s*cn**2
        Ave3 = Ave3 + s*cn**3
        Ave4 = Ave4 + s*cn**4
        #print(s)
    s = E[max_idx] - E[0]
    Ave1 = Ave1/s
    Ave2 = Ave2/s
    Ave3 = Ave3/s
    Ave4 = Ave4/s
    #AveNum = (Ave1)
    return (Ave2 - Ave1**2)
    #VarSig = (Ave4 - 4.*Ave3*AveNum + 8.*Ave2*AveNum**2 - 4.*AveNum**4 - Ave2**2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef  void sigma_array(double[:] Ls, double[:] E, double[:] out):
    cdef int i
    for i in range(Ls.shape[0]):
        out[i] = sigma(Ls[i], E)

def compute_number_variance(Ls, E):
    out = np.empty_like(Ls, dtype='double')
    sigma_array(Ls, E, out)
    return out