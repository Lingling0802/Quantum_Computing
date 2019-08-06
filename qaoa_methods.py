import scipy
import openfermion
import networkx as nx
import os
import numpy as np
import copy
import random
import sys


import operators
from tVQE import *

from openfermion import *


def qaoa(n,
         g,
         adapt_thresh=1e-5,
         theta_thresh=1e-12,
         layer = 1,
         pool=operators.qaoa(),
         ):
    # {{{

    G = g
    pool.init(n, G)
    pool.generate_SparseMatrix()

    hamiltonian = pool.cost_mat[0] * 1j

    w, v = scipy.sparse.linalg.eigs(hamiltonian)
    GS = scipy.sparse.csc_matrix(v[:,w.argmin()]).transpose().conj()
    GS_energy = min(w)

    print('energy:', GS_energy.real)
    print('maxcut objective:', GS_energy.real + pool.shift.real )

    #Start from |+> states: -->
    reference_ket = scipy.sparse.csc_matrix(
        np.full((2**n, 1), 1/np.sqrt(2**n))
    )

    #Start from |1> states: -->
    #ini_0=np.full((2**n - 1, 1), 0)
    #ini_1=np.append(ini_0,[[1]],axis = 0)
    #reference_ket = scipy.sparse.csc_matrix(ini_1)

    reference_bra = reference_ket.transpose().conj()

    # Thetas
    parameters = []

    print(" Start QAOA algorithm")
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  QAOA: ", p+1)
        print(" --------------------------------------------------------------------------")

        ansatz_ops.insert(0, pool.cost_ops[0])
        ansatz_mat.insert(0, pool.cost_mat[0])

        ansatz_ops.insert(0, pool.mixer_ops[0])
        ansatz_mat.insert(0, pool.mixer_mat[0])

        parameters.insert(0, 1)
        parameters.insert(0, 1)

        min_options = {'gtol': theta_thresh, 'disp': False}

        trial_model = tUCCSD(hamiltonian, ansatz_mat, reference_ket, parameters)

        opt_result = scipy.optimize.minimize(trial_model.energy, parameters, jac=trial_model.gradient,
                                             options=min_options, method='Nelder-Mead', callback=trial_model.callback)

        parameters = list(opt_result['x'])
        curr_state = trial_model.prepare_state(parameters)
        print(" Finished: %20.12f" % trial_model.curr_energy)
        print(' maxcut objective:', trial_model.curr_energy + pool.shift)
        print(' Error:', GS_energy.real - trial_model.curr_energy)


