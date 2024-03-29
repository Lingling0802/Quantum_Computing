import openfermion
import numpy as np
import copy as cp

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n = 0
        self.G = 0
        self.w = 0

    def init(self, n, G):

        self.n = n
        self.G = G
        self.w = np.zeros([self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                temp = self.G.get_edge_data(i,j,default=0)
                if temp != 0:
                    self.w[i,j] = temp['weight']
        print(self.w)
        self.generate_SQ_Operators()

    def generate_SparseMatrix(self):
        self.cost_mat = []
        self.mixer_mat = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.cost_ops:
            self.cost_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n))
        for op in self.mixer_ops:
            self.mixer_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n))
            
        return

class qaoa(OperatorPool):
    def generate_SQ_Operators(self):

        A = QubitOperator('Z0 Z1', 0)
        B = QubitOperator('X0', 0)
        C = QubitOperator('Y0', 0)
        D = QubitOperator('Z0 Y1', 0)

        # Cost and mix operators
        self.cost_ops = []
        self.mixer_ops = []


        # Cost Hamiltonian Hp = 0.5*Sum_{i,j}(I-ZiZj)
        self.shift = 0
        for i in range(0,self.n):
            for j in range(i+1,self.n):
                if self.w[i, j] != 0:
                    A += QubitOperator('Z%d Z%d' % (i, j), -0.5j*self.w[i, j])
                    self.shift -= 0.5*self.w[i, j]
        self.cost_ops.append(A)

        # Mixer Hamiltonian Hb = Sum_{i} Xi
        for i in range(0, self.n):
            B += QubitOperator('X%d' % i, 1j)
        self.mixer_ops.append(B)

        # And we can have other choices for mixer operators
        #for i in range(0, self.n):
        #    Y = QubitOperator('Y%d' % i, 1j)

        return






