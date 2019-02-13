import numpy as np

def ecdf(z):
    zz = np.sort(z)
    n = 1 + zz.size
    f = lambda x: np.count_nonzero(zz <= x) / n
    return np.vectorize(f)
    