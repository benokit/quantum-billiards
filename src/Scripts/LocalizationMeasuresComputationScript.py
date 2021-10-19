import sys
sys.path.append("..")
#from ..CoreModules import Weyl as weyl
from ..CoreModules import Utils as ut
from ..CoreModules import HusimiFunctions as hus
import os 

import scipy
import time
import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resource import getrusage, RUSAGE_SELF
from multiprocessing import Pool
#import copy


class parameters:
    def __init__(self, name, ks, idx):
        self.ks = ks
        self.idx = idx
        #self.parameter_string = "name = %s, kind = %s, k1 = %s, k2 = %s, dk = %s, overlap = %s, b = %s, delta = %s" % (name, kind, k1, k2, dk, overlap, b, delta) 
        self.string = r"_%s_idx_%s" % (name,  idx)

class localization_measure_script:
    def __init__(self, make_waf, a, name, kspectrum, measure_exponents = [1,2,4], folder = "", regions=None, grid = None, 
                sym_x = None, sym_y = None, b=3, delta = 5, 
                good_levels = 0.05, n_pieces=50):

        self.kspectrum = kspectrum
        self.make_waf = make_waf
        self.a = a
        self.sym_x = sym_x
        self.sym_y = sym_y
        #self.sym_L = sym_length#length of symmetry axis for weyl formula
        self.measure_exponents = measure_exponents
        self.grid = grid
        self.regions = regions
        if regions is not None:
            gy,gx = regions[0].shape
            self.grid = (gx,gy)

        # numerical parameters
        #self.overlap = overlap
        self.b = b #basis scaling parameter
        self.delta = delta #boundary points scaling parameter
        self.n_pieces = n_pieces
        self.good_levels = good_levels # percentage of expected good levels per diagonalization 
        self.folder = folder
        self.name = name
        self.par_list = [] 
        
        k_pieces = np.array_split(np.array(kspectrum), n_pieces) #parameters are the kspectral pieces
        for i in range(len(k_pieces)):
            par = parameters(name,k_pieces[i],i)
            #print(par)
            self.par_list.append(par)

    def sym_string(self):
        string = ""
        if self.sym_x is not None:
            string = string + "_x_" + self.sym_x
        if self.sym_y is not None:
            string = string + "_y_" + self.sym_y
        return string


    def compute_eigenvectors(self, waf, k, n_l, n_r):
        delta = self.delta
        n_funct = len(waf.basis.basis_functions)
        L = waf.billiard.length
        if waf.scale_basis is not None:
            if not isinstance(waf.scale_basis, list):
                b = np.array([waf.scale_basis for i in range(n_funct)])
                #print("is not list")
                #print(b)
            else:
                b = waf.scale_basis
                #print(b)
            waf.basis.set_basis_size([int(np.ceil(k*L*i/(2*np.pi))) for i in b])
        
        dk = 1
        ks, ten, X = waf.scaling_eigenvectors(k, dk, bnd_pts = None, delta = delta, return_ks = True)
        #X = X*np.sqrt(ten)
        i = (np.abs(ks - k)).argmin()
        ks = ks[i-n_l : i+n_r]
        ten = ten[i-n_l : i+n_r]
        X = X.transpose()
        X = X[i-n_l : i+n_r]
        return ks, ten, X

    def grid_params(self, k, L):#used when grid is variable
        if self.grid is None:
            grd = max(200, int(k/(2*np.pi))) 
            grd_x = int(grd*L)
            grd_y = grd

        else:
            grd_x = self.grid[0]
            grd_y = self.grid[1]
        qs = np.linspace(0, L, grd_x + 1)
        ps  = np.linspace(0, 1, grd_y + 1)
        qs = ut.midpoints(qs) 
        ps = ut.midpoints(ps)
        ncels = grd_x*grd_y
        return qs, ps, ncels

    def compute_pwd(self, par):
        ks = par.ks
        waf = self.make_waf()
        L = waf.billiard.length
        data = []
        start_time = time.time()
        for ki in ks:
            qs, ps, ncels = self.grid_params(ki, L)
            H = waf.Husimi(ki, qs, ps, delta = self.delta)
            data_point = [ki, ki]
            data_point.extend([hus.Renyi_measure(H, a) for a in self.measure_exponents])
            if self.regions is not None:
                region_overlaps = [np.sum(region*H) for region in self.regions]
                data_point.extend(region_overlaps)
            data.append(data_point)
                              
        print("Parameters: %s" % (par.string))
        print("Computation time: %s seconds" % (time.time() - start_time), flush=True)
        column_names = ["old_k", "new_k"]+["l%d"% i for i in self.measure_exponents] 
        if self.regions is not None:
            column_names = column_names +["m%d"% i for i in range(len(self.regions))]
        dataFrame = pd.DataFrame(data, columns = column_names)
        #dataFrame.columns = column_names
        #print(data)
        #print(dataFrame)
        filename = self.folder + r"/measures_piece" + par.string + self.sym_string() +".csv"
        dataFrame.to_csv(filename, header=True, index = False)
        return 0
       

    def compute_sm(self, par):
        ks = par.ks
        waf = self.make_waf()
        L = waf.billiard.length
        
        N = 0 #index of current middle state
        k = ks[N]
        delta_n = max(1,int(k*L/(2*np.pi)*self.good_levels)) #number of good states
        n_l = 0 #states to the left
        n_r = delta_n #states to the right
        data = []
        start_time = time.time()
        state_counter = N
        N1 = len(ks)
        while N  <= N1 + n_r:
            #state_ns = [i for i in range(N-n_l, N+n_r)]

            ks_old = ks[N-n_l:N+n_r]
            ks_new, ten_new, X = self.compute_eigenvectors(waf, k, n_l, n_r)
            #X = X.transpose()
                                
            for i in range(len(ks_new)):
                ki = ks_new[i]
                vec = X[i]
                qs, ps, ncels = self.grid_params(ki, L)
                H = waf.Husimi(ki, qs, ps, delta = self.delta, vec = vec)
                data_point = [ks_old[i], ki]
                data_point.extend([hus.Renyi_measure(H, a) for a in self.measure_exponents])
                if self.regions is not None:
                    region_overlaps = [np.sum(region*H) for region in self.regions]
                    data_point.extend(region_overlaps)
                data.append(data_point)
                state_counter = state_counter + 1
                if state_counter == N1:
                    break
            N = N + 2*n_r
            n_l = n_r
            if N >= N1:
                k = ks[N-n_r]
                n_l = 0
                n_r = abs(N1 - (N-n_r))
            else:
                k = ks[N]
                delta_n = max(1,int(k*L/(2*np.pi)*self.good_levels)) #number of good states
                #print(delta_n)
                n_r = delta_n
                
        print("Parameters: %s" % (par.string))
        print("Computation time: %s seconds" % (time.time() - start_time), flush=True)
        column_names = ["old_k", "new_k"]+["l%d"% i for i in self.measure_exponents]
        if self.regions is not None:
            column_names = column_names +["m%d"% i for i in range(len(self.regions))] 
        #print(column_names)
        dataFrame = pd.DataFrame(data, columns = column_names)
        #dataFrame.columns = column_names
        #print(data)
        #print(dataFrame)
        filename = self.folder + r"/measures_piece" + par.string + self.sym_string() +".csv"
        dataFrame.to_csv(filename, header=True, index = False)
        return 0

    def load_measures_piece(self, par):
        filename = self.folder + r"/measures_piece" + par.string + self.sym_string() + ".csv"
        #if os.path.getsize(filename) == 0:
            #print('File is empty')
            #return np.array([]), np.array([])
        data = pd.read_csv(filename, header=0)
        #ks, ten = np.array(data).transpose()
        return data

    def combine_spectral_pieces(self, delete_pieces = False):

        string = r"_%s" % (self.name)
        data_frames = []
        for par in self.par_list:
            #print(par.string)
            data = self.load_measures_piece(par)
            #all_ks = np.concatenate([all_ks, ks])
            data_frames.append(data)
        dataFrame = pd.concat(data_frames)
        filename = self.folder + r"/measures" + string + self.sym_string() + ".csv"
        dataFrame.to_csv(filename, header=True, index = False)

        if delete_pieces:
            for par in self.par_list:
                filename = self.folder + r"/measures_piece" + par.string + self.sym_string() + ".csv"
                if os.path.isfile(filename):
                    os.remove(filename)
                else:    ## Show an error ##
                    print("Error: %s file not found" % filename)
            print("Directory cleaned!")


    def run_script(self, n_processes, method = "pwd",  delete_pieces = False):
        if method == "pwd":
            start = time.time()
            with Pool(processes = n_processes) as pool:
                results = list(pool.apply_async(self.compute_pwd, args=(i, )) for i in self.par_list)
                results = [r.get() for r in results]
            end = time.time()
            print('Final time: %s sec' % (end - start))
            print("Peak Memory usage: %s GB" % (getrusage(RUSAGE_SELF).ru_maxrss/1024/1024))
            self.combine_spectral_pieces(delete_pieces = delete_pieces)
        elif method == "sm":
            start = time.time()
            with Pool(processes = n_processes) as pool:
                results = list(pool.apply_async(self.compute_sm, args=(i, )) for i in self.par_list)
                results = [r.get() for r in results]
            end = time.time()
            print('Final time: %s sec' % (end - start))
            print("Peak Memory usage: %s GB" % (getrusage(RUSAGE_SELF).ru_maxrss/1024/1024))
            self.combine_spectral_pieces(delete_pieces = delete_pieces)        