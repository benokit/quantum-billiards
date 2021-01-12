import numpy as np

def unfold(billiard, E):
    L = billiard.length
    A = billiard.area
    unfolded = (A*E - L*np.sqrt(E))/(4*np.pi)
    return unfolded

def k_at_state(billiard, state):
    L = billiard.length
    A = billiard.area
    a = A
    b = -L
    c = -state*4*np.pi
    dis = np.sqrt(b**2-4*a*c)
    return (-b+dis)/(2*a)

def state_at_k(billiard, k):
    L = billiard.length
    A = billiard.area
    return (A*k**2 - L*k)/(4*np.pi)

def count_states(z):
    zz = np.sort(z)
    f = lambda x: np.count_nonzero(zz <= x)
    return np.vectorize(f)

def check_states(billiard, ks, grid = 10):
    count_fun = count_states(ks) #defines function
    kmin = np.min(ks)
    kmax = np.max(ks)
    x = np.linspace(kmin, kmax, grid)
    n = count_fun(x) #number of states < x
    N = unfold(billiard, ks**2)
    return x, n-N