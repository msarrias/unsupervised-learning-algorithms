import numpy as np
import random
from collections import Counter
from matplotlib import pyplot as plt
import networkx as nx
import scipy.linalg as la

class LE:   
    def __init__(self, par, nn_graph, m, coord, sim_measure = 'inv_ed' , print_ = False):
        self.sim_measure = sim_measure
        if self.sim_measure == 'inv_ed':
            self.k = par
        if self.sim_measure == 'gk':
            self.k = par[0]
            self.t = par[1]
        self.m = m
        self.coord = coord
        self.N = len(coord)
        self.G = self.init_graph_nodes()
        if print_: print(f'- Initialized {nn_graph} graph.')
        self.add_nodes_position()
        if print_: print('- Added nodes.')
        self.eucledian_distance_matrix()
        if print_: print('- Computed the Eucledian Distance similarity matrix.')
        if nn_graph == 'knn':
            self.connect_nodes_knn()
        if nn_graph == 'mutual_knn':
            self.connect_nodes_mutual_nn()
        if print_: print(f'- Added edges and weights to each node {self.k} {nn_graph}')
        if print_: print(f'and weights given by the heat kernel with t = {self.t}')
        if print_: print(f'- Single component Graph: {nx.is_connected(self.G)}')
        self.A = self.adjacency_matrix()
        self.Vol = np.sum(self.A)
        self.sqrtVol = np.sqrt(self.Vol)
        self.D = self.diagonal_matrix()
        self.L_unnorm = self.D - self.A
        self.L_symm = np.matmul(np.matmul(np.diag(sum(self.D)**(-1/2)), self.L_unnorm), 
                                np.diag(sum(self.D)**(-1/2)))
        if print_: print(f'- Computed the unnormalized Laplacian')
        self.solve_eigenvalue_problem()
        if print_: print(f'- Solved the generalized eigenvalue problem')
        self.commute_time_embedding()

    def init_graph_nodes(self):
        G = nx.Graph()
        G.add_nodes_from(range(1, self.N + 1 ))
        return G
        
    def add_nodes_position(self):
        pos_dic = {i :self.coord[i-1] for i in range(1, self.N + 1)} 
        for n, p in pos_dic.items():
            self.G.nodes[n]['pos'] = p
    
    def eucledian_dist(self, x_i, x_j):
        return np.sqrt(sum((x_i - x_j)**2))

    def eucledian_distance_matrix(self):
        self.distance_matrix = np.zeros((self.N, self.N))
        for xi in range(self.N):
            for xj in range(xi, self.N):
                self.distance_matrix[xi, xj] = self.eucledian_dist(self.coord[xi], 
                                                                   self.coord[xj])
                self.distance_matrix[xj, xi] = self.distance_matrix[xi, xj]
    
    def heat_kernel(self, eucled_dist):
        if self.t != np.infty:
            return np.exp(- eucled_dist**2 / self.t)
        else:
            return 1

    def connect_nodes_mutual_nn(self):
        for i in range(self.N):
            for link in self.distance_matrix[i].argsort()[1:self.k]:
                if i in self.distance_matrix[link].argsort()[1:self.k]:
                    self.G.add_edge(i+1, link+1, 
                                    weight = self.heat_kernel(self.distance_matrix[i][link]))
                
    def connect_nodes_knn(self):
        if self.sim_measure == 'gk':
            for i in range(self.N):
                for link in self.distance_matrix[i].argsort()[1:self.k]:
                    self.G.add_edge(i+1, link+1, 
                                    weight = self.heat_kernel(self.distance_matrix[i][link]))
        
        if self.sim_measure == 'inv_ed':
            for i in range(self.N):
                for link in self.distance_matrix[i].argsort()[1:self.k]:
                    self.G.add_edge(i+1, link+1, 
                                    weight = 1/self.distance_matrix[i][link])
            

    def adjacency_matrix(self):
        return nx.adjacency_matrix(self.G).todense()
    
    def diagonal_matrix(self):
        return np.diag(np.asarray(sum(self.A))[0])
    
    def unnormalized_Laplacian(self):
        return self.D - self.A
    
    def solve_eigenvalue_problem(self):
        # Solve standard Eigenvalue problem of L:
        eigenvalues_l, eigenvectors_l= la.eig(self.L_unnorm)
        sorted_eigenv_l =  sorted(zip(eigenvalues_l.real, eigenvectors_l.T), key=lambda x: x[0])
        self.eigenvalues_l = [elm[0] for elm in sorted_eigenv_l] 
        self.eigenvectors_l = [elm[1] for elm in sorted_eigenv_l]
        self.Vl_T = np.vstack(self.eigenvectors_l[1:])
        
        # Solve generalized Eigenvalue problem of L to get the eigenvalues and eigenvectors of L_rw:
        eigenvalues_lrw, eigenvectors_lrw = la.eig(self.L_unnorm, self.D)
        sorted_eigenv_lrw =  sorted(zip(eigenvalues_lrw.real, eigenvectors_lrw.T), key=lambda x: x[0])
        self.eigenvalues_lrw = [elm[0] for elm in sorted_eigenv_lrw] 
        self.eigenvectors_lrw = [elm[1] for elm in sorted_eigenv_lrw]
        
        # Get eigenvectors of L_sym from L_rw: 
        self.eigenvalues_lsym = self.eigenvalues_lrw
        self.eigenvectors_lsym = [np.dot(np.sqrt(self.D),
                                    elm[1])/la.norm(np.dot(np.sqrt(self.D),
                                                           elm[1])) for elm in sorted_eigenv_lrw]
        #observations are column wise m x n
        self.V_T = np.vstack(self.eigenvectors_lsym[1:]) #rows: eigenvectors
        
    def commute_time_embedding(self):
        # CTE in terms of the normalized Laplacian L_sym
        sqrt_inv_A = np.diag(1 / np.sqrt(np.asarray(self.eigenvalues_lsym[1:]) + 1e-10))
        sqrt_inv_D = np.diag(1 / np.sqrt(np.asarray(sum(self.A))[0]))
        LHS = self.sqrtVol * sqrt_inv_A[0:self.m, 0:self.m]
        self.CTE = np.dot(LHS, self.V_T[0:self.m, :]).dot(sqrt_inv_D)
        
        # CTE in terms of the unnormalized Laplacian L
        sqrt_inv_Al = np.diag(1 / np.sqrt(np.asarray(self.eigenvalues_l[1:]) + 1e-10))
        LHS_l = self.sqrtVol * sqrt_inv_Al[0:self.m, 0:self.m]
        self.CTEl = np.dot(LHS_l, self.Vl_T[0:self.m, :])



