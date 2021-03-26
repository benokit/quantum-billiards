import numpy as np

def unfold(billiard, E, sym_length = None):
    L = billiard.length
    if sym_length is not None:
        L = L + sym_length
    A = billiard.area
    unfolded = (A*E - L*np.sqrt(E))/(4*np.pi)
    return unfolded

def k_at_state(billiard, state, sym_length = None):
    L = billiard.length
    if sym_length is not None:
        L = L + sym_length
    A = billiard.area
    a = A
    b = -L
    c = -state*4*np.pi
    dis = np.sqrt(b**2-4*a*c)
    return (-b+dis)/(2*a)

def state_at_k(billiard, k, sym_length = None):
    L = billiard.length
    if sym_length is not None:
        L = L + sym_length
    A = billiard.area
    return (A*k**2 - L*k)/(4*np.pi)

def count_states(z):
    zz = np.sort(z)
    f = lambda x: np.count_nonzero(zz <= x)
    return np.vectorize(f)

def check_states(billiard, ks, sym_length = None, grid = 10):
    count_fun = count_states(ks) #defines function
    kmin = np.min(ks)
    kmax = np.max(ks)
    x = np.linspace(kmin, kmax, grid)
    n = count_fun(x) #number of states < x
    N = unfold(billiard, x**2, sym_length =sym_length) - unfold(billiard, x[0]**2, sym_length =sym_length)
    return x, n-N

def mean_level_spacing(billiard, n, d=10, sym_length = None):
    k = k_at_state(billiard, n, sym_length = None)
    dk = np.mean(np.diff(k_at_state(billiard, np.linspace(n, n+d, d), sym_length = None)))
    return k, dk