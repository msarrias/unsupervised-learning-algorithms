import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from kmeans import *

class FLAME:
    def __init__(self, data_coord, k, outlier_threshold, eps = 1e-10, verbose = False):
        
        self.data_coord = data_coord # N x M
        self.N, _ = self.data_coord.shape
        self.k_max = int(np.sqrt(self.N) + 10)
        print(self.k_max)
        self.outlier_threshold = outlier_threshold 
        self.eps = eps
        self.verbose = verbose
        
        if k > self.k_max: self.k = self.k_max 
        else: self.k = k # nearest neighbors
    
        self.objects_nn_distances = {}
        self.objects_nn_ids = {}
        self.W = {}
        self.objects_density = {}

    def eucledian_dist(self, x_i, x_j):
        return np.sqrt(sum((x_i - x_j)**2))

    def eucledian_distance_matrix(self):
        self.distance_matrix = np.zeros((self.N, self.N))
        for xi in range(self.N):
            for xj in range(xi, self.N):
                self.distance_matrix[xi, xj] = self.eucledian_dist(self.data_coord[xi], 
                                                                   self.data_coord[xj])
                self.distance_matrix[xj, xi] = self.distance_matrix[xi, xj]
       
    def compute_object_density(self, i):
        return(self.max_dist / (np.sum(self.objects_nn_distances[i]) / len(self.objects_nn_distances[i])))
    
    def compute_object_weight(self, k_temp):
        return( {j: (k_temp - j) / (0.5 * k_temp * (k_temp + 1.0)) for j in range(k_temp)})
    
    def classify_objects(self):
        #Density threshold for possible outliers
        objects_dens = np.asarray(list(self.objects_density.values()))
        mean_dens = np.mean(list(self.objects_density.values()))
        RHS = self.outlier_threshold * np.sqrt(np.dot(objects_dens, objects_dens) / self.N - mean_dens * mean_dens)
        self.outlier_threshold = mean_dens + RHS
        print(self.outlier_threshold)
        #objects with density higher than all his neighbors
        self.CSO = []
        #object with density lower than all its neighbors, and lower than a threshold
        self.outliers = []
        #the rest
        self.rest = []
        for i in range(self.N):
            temp_k = len(self.objects_nn_distances[i])
            temp = [self.objects_density[j] for j in self.knn_ids_matrix[i][0 : temp_k]]
            if self.objects_density[i] > max(temp):
                self.CSO.append(i)
            elif self.objects_density[i] < min(temp) and self.objects_density[i] <= self.outlier_threshold:
                self.outliers.append(i)
            else:
                self.rest.append(i)
        self.CSO_N = len(self.CSO)
        #|X_CSO| + outliers group
        self.M = self.CSO_N + 1
        
    def initialize_fuzzy_membership(self):
        self.fuzzyship_matrix = np.zeros((self.N, self.M))
        self.init_fuzzyship_matrix = np.zeros((self.N, self.M))
        #Each CSO is assigned with fixed and full membership to itself to represent one cluster;
        for k, cso in enumerate(self.CSO):
            self.fuzzyship_matrix[cso][k] = 1.0
            self.init_fuzzyship_matrix[cso][k] = 1.0
        #All outliers are assigned with fixed and full membership to the outlier group;
        for o in self.outliers:
            self.fuzzyship_matrix[o][self.M - 1] = 1.0
            self.init_fuzzyship_matrix[o][self.M - 1] = 1.0
        #The rest are assigned with equal memberships to all clusters and the outlier group;
        for r in self.rest:
            self.fuzzyship_matrix[r][0 : self.M] = 1.0 / (self.M)
            self.init_fuzzyship_matrix[r][0 : self.M] = 1.0 / (self.M)
        
    def update_fuzzy_membership(self, it):
        local_neighb_approx_error = 0
        for xi in range(self.N):
            if xi in self.rest:
                knn_xi = len(self.objects_nn_distances[xi])
                xi_ids = list(self.objects_nn_ids[xi])
                xi_weights = np.asarray(list(self.W[xi].values()))
                for k in range(self.M):
                    if(it % 2 == 0):
                        self.fuzzyship_matrix[xi][k] = np.dot(xi_weights, 
                                                              self.init_fuzzyship_matrix[xi_ids, k])
                        temp = np.dot(xi_weights, self.fuzzyship_matrix[xi_ids, k])
                    else:
                        self.init_fuzzyship_matrix[xi][k] = np.dot(xi_weights, 
                                                                   self.fuzzyship_matrix[xi_ids, k])
                        temp = np.dot(xi_weights, self.fuzzyship_matrix[xi_ids, k])

                    local_neighb_approx_error += np.dot(self.fuzzyship_matrix[xi][k] - temp,
                                                        self.fuzzyship_matrix[xi][k] - temp)
        return(local_neighb_approx_error)
    
    def construct_clusters(self):
        self.clusters_pred = np.asarray([0] * self.N)
        self.clusters_pred[self.CSO] = [cso for cso in range(self.CSO_N)]
        self.clusters_pred[self.rest] = self.fuzzyship_matrix[self.rest].argsort()[:,-1]
        self.clusters_pred[self.outliers] = self.M - 1
        
        
    def fit_FLAME(self):
        #-----------------  FIRST PART: Extraction of the structure information -----------------#
        #pairwise distance measure
        self.eucledian_distance_matrix() 
        #get objects k nearest neighbors idx
        self.knn_ids_matrix = self.distance_matrix.argsort()[:, 1:]
        #sort distance matrix to get the knn distances
        self.distance_matrix.sort()
        #do not include self-distance
        self.distance_matrix = self.distance_matrix[:, 1:]
        self.max_dist = np.max(self.distance_matrix)
        # include those neighbours with same distance as the neighbor with the higher distance.
        for i in range(self.N):
            k_temp = self.k
            dist_most_distant_neighb = self.distance_matrix[i][self.k-1]
            for j in range(self.k, self.k_max):
                if self.distance_matrix[i][j] == dist_most_distant_neighb:
                    k_temp += 1
                break
            #not all objets will have knn
            self.objects_nn_distances[i] = self.distance_matrix[i, 0:k_temp]
            self.objects_nn_ids[i] = self.knn_ids_matrix[i][0 : k_temp]
            #compute objects density:
            self.objects_density[i] = self.compute_object_density(i)
            #compute objects weights:
            self.W[i] = self.compute_object_weight(k_temp)
        #Classify objects:
        self.classify_objects()
        if self.verbose: 
            print(f'number of CSO: {self.CSO_N}, outliers {len(self.outliers)}')
            print(f'and rest: {len(self.rest)}')
        #-----------------  SECOND PART: Local/Neigb. approximation of fuzzy memberships --------#
        #Initialization of fuzzy membership
        self.initialize_fuzzy_membership()
        #Update of fuzzy membership
        local_neighb_approx_error = np.infty
        it = 0
        if self.verbose: print('it:')
        while local_neighb_approx_error > self.eps:
            local_neighb_approx_error = self.update_fuzzy_membership(it)
            if self.verbose: 
                if it % 5 == 0: print(it, end = '--')
            it += 1  
        if self.verbose: print(it)
        #-----------------  THIRD PART: Cluster construction from fuzzy memberships ------------#
        #One-to-one object-cluster assignment, 
        #to assign each object to the cluster in which it has the highest membership
        self.construct_clusters()
        