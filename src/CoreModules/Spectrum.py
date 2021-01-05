import numpy as np
from scipy import optimize
from . import Utils as ut
from . import Solvers as solvers
import matplotlib.pyplot as plt


def glueSpectra(leftSpectrum, rightSpectrum, leftTension, rightTension, tolerance = None):
    """Determines the overlaping levels between two spectral intervals 
    and merges the two spectra by keeping the most accurate of the overlaping levels.  
    
    This function is used in the spectrum class.

    Parameters
    ----------
    leftSpectrum : numpy array
        An array of eigenenergies or eigenwavenumbers 
        in the lower spectral interval.
    rightSpectrum : numpy array
        An array of eigenenergies or eigenwavenumbers 
        in the higher spectral interval.
    leftTension : numpy array
        An array of tensions characterising the eigenfunctions 
        in the lower spectral interval.
    rightTension : numpy array
        An array of tensions characterising the eigenfunctions 
        in the higher spectral interval.
    
    Returns
    -------
    final_spec : numpy array
        The merged spectrum.
    final_tension : numpy array
        The merged tensions.    
    """
    sz_l = leftSpectrum.size
    sz_r = rightSpectrum.size
    #print("initial")
    #print(leftSpectrum)
    #print(rightSpectrum)

    #check if spectra are empty
    if (sz_r == 0):
        return leftSpectrum, leftTension
    if (sz_l == 0):
        return rightSpectrum, rightTension
    #find k overlap
    k_end_l = leftSpectrum[-1] #rightmost point in leftSpectrum
    #print(k_end_l)
    end_idx_r = np.argmin(rightSpectrum <= k_end_l)
    k_start_r = rightSpectrum[0] #leftmost point in rightSpectrum
    #print(k_start_r)
    if tolerance is not None:
        k_end_l = k_end_l + tolerance
        k_start_r = k_start_r - tolerance
    if k_end_l < k_start_r:
        #print("No overlap")
        return np.concatenate((leftSpectrum,rightSpectrum)), np.concatenate((leftTension,rightTension))

    start_idx_l = np.argmax(leftSpectrum >= k_start_r)
    o_left = leftSpectrum[start_idx_l : ] #ovelaping levels of leftSpectrum
    o_right = rightSpectrum[ : end_idx_r] #ovelaping levels of rightSpectrum
    o_ten_left = leftTension[start_idx_l : ] #ovelaping tensions of leftSpectrum
    o_ten_right = rightTension[ : end_idx_r] #ovelaping tensions of rightSpectrum

    n_l = o_left.size
    n_r = o_right.size
    if n_r > n_l:
        o_right = np.copy(o_right[n_r-n_l : ])
        o_ten_right = np.copy(o_ten_right[n_r-n_l : ])
        #print("case 1")
        #print(n_r-n_l)
        #print(o_left)
        #print(o_right)    

    if n_r < n_l:
        
        o_left = np.copy(o_left[ : -(n_l-n_r) ])
        o_ten_left = np.copy(o_ten_left[:-(n_l-n_r)])
        #print("case 2")
        #print(n_l-n_r)
        #print(o_left)
        #print(o_right)
    #else:
        #print("case 0")
        #print(o_left)
        #print(o_right)
    idx = o_ten_left > o_ten_right #find indices of smaller tension
    o_spectrum = o_left
    o_spectrum[idx] = o_right[idx]
    o_tension = o_ten_left
    o_tension[idx] = o_ten_right[idx]


    #print("chuncks")
    #print(leftSpectrum[ : start_idx_l])
    #print(o_spectrum)
    #print(rightSpectrum[end_idx_r: ])
    final_spec = np.concatenate((leftSpectrum[ : start_idx_l], o_spectrum, rightSpectrum[end_idx_r: ]))
    final_tension = np.concatenate((leftTension[ : start_idx_l], o_tension, rightTension[end_idx_r: ]))
    #print("final")
    #Sprint(final_spec)
    return final_spec, final_tension


class spectrum:
    """ A class used to represent the billiard eigenvalue problem and compute the spectrum.

    Attributes
    ----------
    billiard : billiard object
        A billiard object that represents the billiard table.
    basis : basis object
        A basis object that represents the basis of the Hilbert space.

    Methods
    -------
    compute_k(k0, dk, solver = "DM", point_density = 100, Mi= 100,
                scale_basis = None, eps = 0.5e-15 )

    plot_tension(kmin, kmax, solver = "DM", point_density = 100, Mi= 100,
                 grid = 200, plot_params = {}, log = True, scale_basis = None, eps = 0.5e-15)

    compute_spectrum(self, k1, k2, dk, overlap = 0.3, point_density = 100, scale_basis = None, 
                    eps = 0.5e-15, tolerance=None, return_tensions = True)

    test_sm(self, k0 ,dk, point_density = 100, scale_basis = None, eps = 0.5e-15)


    correct_spectrum(self, ks, dk, solver= "DM", point_density = 100, Mi= 100,
                     scale_basis = None, eps = False, return_tensions = True)
    """

    def __init__(self, billiard, basis):
        self.billiard = billiard
        self.basis = basis

    def compute_k(self, k0, dk, solver = "DM", point_density = 100, Mi= 100, scale_basis = None, eps = 0.5e-15 ):
        #adjust basis size
        L = self.billiard.length
        A = self.billiard.area
        n_funct = len(self.basis.basis_functions)
        if scale_basis is not None:
            if not isinstance(scale_basis, list):
                b = np.array([scale_basis for i in range(n_funct)])
                #print("is not list")
                #print(b)
            else:
                b = scale_basis
                #print(b)
            self.basis.set_basis_size([int(np.ceil(k0*L*i/(2*np.pi))) for i in b])
        
        bnd_pts = self.billiard.evaluate_boundary(point_density, evaluate_virtual = False,  midpts = True, normal = True, weights = True)

        if solver == "DM":
            res = optimize.minimize_scalar(lambda k: solvers.decomposition_method
                                                    (k, self.basis, bnd_pts, L, A, eps = eps), 
                                                    bounds=(k0-dk, k0+dk), method='bounded')
            return res.x, res.fun
        
        if solver == "SM":
            ks, tensions = solvers.scaling_method(k0, dk,  self.basis, bnd_pts, eps = eps)
            idx = (np.abs(ks - k0)).argmin()
            return ks[idx], tensions[idx]
        
        if solver == "PSM":
            int_pts = self.billiard.random_interior_points(Mi)
            res = optimize.minimize_scalar(lambda k: solvers.particular_solutions_method
                                                    (k, self.basis, bnd_pts, int_pts, eps = eps),
                                                     bounds=(k0-dk, k0+dk), method='bounded')
            return res.x, res.fun
        else:
            return 0, 0

    def plot_tension(self, kmin, kmax, solver = "DM", point_density = 100, Mi= 100, grid = 200, plot_params = {}, log = True, scale_basis = None, eps = 0.5e-15):
                
        bnd_pts = self.billiard.evaluate_boundary(point_density, evaluate_virtual = False, midpts = True, normal = True, weights = True)
        
        L = self.billiard.length
        n_funct = len(self.basis.basis_functions)
        k0 = kmax
        if scale_basis is not None:
            if not isinstance(scale_basis, list):
                b = np.array([scale_basis for i in range(n_funct)])
                #print("is not list")
                #print(b)
            else:
                b = scale_basis
                #print(b)
            self.basis.set_basis_size([int(np.ceil(k0*L*i/(2*np.pi))) for i in b])

        if solver == "DM":
            L = self.billiard.length
            A = self.billiard.area
            k_vals = np.linspace(kmin, kmax, grid)
            tensions = [solvers.decomposition_method(k, self.basis, bnd_pts, L, A, eps = eps) for k in k_vals]
            if log:
                plt.semilogy(k_vals,tensions,**plot_params)
            else:
                plt.plot(k_vals,tensions, **plot_params)
            plt.xlabel("$k$")
            return k_vals, tensions
        
        if solver == "PSM":
            k_vals = np.linspace(kmin, kmax, grid)
            int_pts = self.billiard.random_interior_points(Mi)
            tensions = [solvers.particular_solutions_method(k, self.basis, bnd_pts, int_pts, eps = eps) for k in k_vals]
            if log:
                plt.semilogy(k_vals,tensions, **plot_params)
            else:
                plt.plot(k_vals,tensions, **plot_params)
            plt.xlabel("$k$")
            return k_vals, tensions
        
        else:
            print("Solver is wrong, select from: ")
            return 0


    def compute_spectrum(self, k1, k2, dk, overlap = 0.3, point_density = 100, scale_basis = None, eps = 0.5e-15, tolerance=None, return_tensions = True):
        
        bnd_pts = self.billiard.evaluate_boundary(point_density, evaluate_virtual = False, midpts = True, normal = True, weights = True)
        
        L = self.billiard.length
        n_funct = len(self.basis.basis_functions)
        if scale_basis is not None:
            if not isinstance(scale_basis, list):
                b = np.array([scale_basis for i in range(n_funct)])
                #print("is not list")
                #print(b)
            else:
                b = scale_basis
                #print(b)
            self.basis.set_basis_size([int(np.ceil(k2*L*i/(2*np.pi))) for i in b])

        spect0, tensions = solvers.scaling_method(k1, dk*(1+overlap),  self.basis, bnd_pts, eps = eps)
        spect = spect0[spect0 > k1]
        tensions = tensions[spect0 > k1]
        #print(len(spect),len(tensions))
        k = k1 + 2*dk
        while (k <= k2 + 2*dk): 
            #print(spect)           
            
            #print     
            s, t = solvers.scaling_method(k, dk*(1+overlap), self.basis, bnd_pts, eps = eps)
            k = k + 2*dk
            #print(s)
            #print(t)
            #print(len(s),len(t))
            #print(s)
            spect, tensions= glueSpectra(spect, s, tensions, t ,tolerance=tolerance)
        if return_tensions:
            return spect[spect < k2], tensions[spect < k2]
        else:
            return spect[spect < k2]

    def test_sm(self, k0 ,dk, point_density = 100, scale_basis = None, eps = 0.5e-15):
        bnd_pts = self.billiard.evaluate_boundary(point_density, evaluate_virtual = False, midpts = True, normal = True, weights = True)
        k2 = k0
        L = self.billiard.length
        n_funct = len(self.basis.basis_functions)
        if scale_basis is not None:
            if not isinstance(scale_basis, list):
                b = np.array([scale_basis for i in range(n_funct)])
                #print("is not list")
                #print(b)
            else:
                b = scale_basis
                #print(b)
            self.basis.set_basis_size([int(np.ceil(k2*L*i/(2*np.pi))) for i in b])

        return solvers.scaling_method(k0, dk,  self.basis, bnd_pts, eps = eps)



    def correct_spectrum(self, ks, dk, solver= "DM", point_density = 100, Mi= 100 , scale_basis = None, eps = False, return_tensions = True):
        res = [self.compute_k(k0, dk, solver = solver, point_density = point_density,
                            Mi= Mi, scale_basis = scale_basis, eps = eps) for k0 in ks]
        s, t = np.transpose(np.array(res))
        return s, t

