import sys
sys.path.append("..")
from ..CoreModules import Weyl as weyl
import os 

import scipy
import time
import math as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from resource import getrusage, RUSAGE_SELF
from multiprocessing import Pool
import copy





class parameters:
    def __init__(self, name, n0, dn, k1, k2, dk, idx):
        self.k1 = k1
        self.k2 = k2
        self.dk = dk
        self.idx = idx
        #self.parameter_string = "name = %s, kind = %s, k1 = %s, k2 = %s, dk = %s, overlap = %s, b = %s, delta = %s" % (name, kind, k1, k2, dk, overlap, b, delta) 
        self.string = r"_%s_n_%d_to_%d_idx_%s" % (name, n0, n0+dn, idx)

class billiard_spectrum_script:
    def __init__(self, make_evp, a, name, folder, n0, dn, sym_x = None, sym_y = None, sym_length=None, b=3, delta = 5, overlap = 0.2, good_levels = 0.05, n_steps=200):
        self.make_evp = make_evp
        self.a = a
        self.sym_x = sym_x
        self.sym_y = sym_y
        self.sym_L = sym_length#length of symmetry axis for weyl formula
        self.n0 = n0 #start state
        self.dn = dn #number of states computed
        # numerical parameters
        self.overlap = overlap
        self.b = b #basis scaling parameter
        self.delta = delta #boundary points scaling parameter
        self.n_steps = n_steps
        self.good_levels = good_levels # percentage of expected good levels per diagonalization 
        self.folder = folder
        self.name = name
        self.par_list = []

    def sym_string(self):
        string = ""
        if self.sym_x is not None:
            string = string + "_x_" + self.sym_x
        if self.sym_y is not None:
            string = string + "_y_" + self.sym_y
        return string

    def mean_level_spacing(self, evp, n,  d=100):
        k = weyl.k_at_state(evp.billiard, n, sym_length = self.sym_L)
        dk = np.mean(np.diff(weyl.k_at_state(evp.billiard, np.linspace(n, n+d, d), sym_length=self.sym_L)))
        return k, dk


    def build_parameter_list(self, evp):
        n_steps = self.n_steps
        k1 = weyl.k_at_state(evp.billiard, self.n0)
        k2 = weyl.k_at_state(evp.billiard, self.n0 + self.dn)
        #print(k1)
        #print(k2)
        k_mls, mls = self.mean_level_spacing(evp, self.n0)
        L = evp.billiard.length 
        delta_n = max(1,int(k1*L/(2*np.pi)*self.good_levels))
        dk = delta_n*mls*1
        #print(delta_n)
        
        k = k1-dk
        i = 0
        while k<k2+dk:
            par = parameters(self.name, self.n0, self.dn, k, k + dk*n_steps, dk, i)
            self.par_list.append(par)
            k = k + dk*n_steps
            n_i = weyl.state_at_k(evp.billiard,k)
            k_mls, mls = self.mean_level_spacing(evp, n_i)
            delta_n = max(1,int(k*L/(2*np.pi)*self.good_levels)) #estimated number of good states per diagonalisation
            dk = delta_n*mls
            #print("interval %s" %i)
            #print(delta_n)
            #print(dk)
            i = i + 1
        
    def compute(self, par):

        k1 = par.k1
        k2 = par.k2
        ov = self.overlap
        b = self.b
        delta = self.delta
        dk =  par.dk   
        evp = self.make_evp()
        start_time = time.time()
        data = evp.compute_spectrum(k1, k2, dk, overlap = ov, delta = delta, scale_basis = b, print_info = False)
        print("\n")
        print("Parameters: %s" % (par.string))
        print("Piece computation time: %s seconds" % (time.time() - start_time), flush=True)

        dataFrame = pd.DataFrame(np.transpose(data))

        filename = self.folder + r"/spectral_piece" + par.string + self.sym_string() +".csv"
        dataFrame.to_csv(filename, header=False, index = False)

    def load_spectral_piece(self, par):
        filename = self.folder + r"/spectral_piece" + par.string + self.sym_string() + ".csv"
        if os.path.getsize(filename) == 0:
            print('File is empty')
            return np.array([]), np.array([])
        data = pd.read_csv(filename, header=None)
        ks, ten = np.array(data).transpose()
        return ks, ten

    def combine_spectral_pieces(self, delete_pieces = False):
        all_ks = []
        all_ten = []
                     
        string = r"_%s_n_%d_to_%d" % (self.name, self.n0, self.n0+self.dn)
        
        for par in self.par_list:
            #print(par.string)
            ks, ten = self.load_spectral_piece(par)
            all_ks = np.concatenate([all_ks, ks])
            all_ten = np.concatenate([all_ten, ten])
                
        data = all_ks, all_ten
        dataFrame = pd.DataFrame(np.transpose(data))

        filename = self.folder + r"/kspectrum" + string + self.sym_string() + ".csv"
        dataFrame.to_csv(filename, header=False, index = False)

        if delete_pieces:
            for par in self.par_list:
                filename = self.folder + r"/spectral_piece" + par.string + self.sym_string() + ".csv"
                if os.path.isfile(filename):
                    os.remove(filename)
                else:    ## Show an error ##
                    print("Error: %s file not found" % filename)
            print("Directory cleaned!")
    #return all_ks, all_ten

    def filter_parameters(self, par_list):
        def check_if_exists(par):
            filename = self.folder + r"/spectral_piece" + par.string + self.sym_string() + ".csv"
            return os.path.exists(filename)
        
        check = [check_if_exists(par) for par in par_list]
        filtered_list = []
        for i in range(len(check)):
            if not check[i]:
                filtered_list.append(par_list[i])
        self.par_list = filtered_list




    def run_script(self, n_processes, filter_pieces=False, delete_pieces = False):
        evp = self.make_evp()
        self.build_parameter_list(evp)
        
        #pool = Pool(processes = n_processes)
        start = time.time()
        #pool.map(self.compute, self.par_list)
        with Pool(processes = n_processes) as pool:
            results = list(pool.apply_async(self.compute, args=(i, )) for i in self.par_list)
            results = [r.get() for r in results]
        end = time.time()
        print('Final time: %s sec' % (end - start))
        print("Peak Memory usage: %s GB" % (getrusage(RUSAGE_SELF).ru_maxrss/1024/1024))
        self.combine_spectral_pieces(delete_pieces = delete_pieces)

