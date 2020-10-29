import numpy as np
import random

class Kmeans:
    
    def __init__(self, K, eps, seed):
        
        self.K = K
        self.eps = eps
        self.seed = seed
        
    def initial_centroids(self, X):
        """
        initial_centroids is a function that initialices the 
        centroides by randomly selecting k D-dimensional 
        data points from our dataset X.
        :return: K * D matrix
        """
        random.seed(self.seed)
        idxs = random.sample(range(len(X)), self.K)
        centroids = np.asanyarray([X[i] for i in idxs])
        return centroids

    def compute_centroids(self, X, labels):
        """
        compute_centroids is a function that re-computes the 
        centroids as the mean of all the data points in X
        assigned to cluster k.
        :return: K * D matrix
        """
        centroids = np.zeros(self.centroids.shape)
        for k in range(self.K):
            centroids[k] = np.mean(X[np.where(labels == k)], 0)
        return centroids 

    def square_distance(self, x_i, x_j):
        """
        square_distance is a function that computes the sum of 
        the squares of the distances of each data point to the 
        k-th centroid.
        :return: scalar
        """
        n_coord = x_i.shape[0]
        square_distance = 0
        if n_coord == x_j.shape[0]:
            for i in range(n_coord):
                square_distance += (x_i[i] - x_j[i])**2
        return square_distance

    def distance_to_centroids(self, X, centroids):
        """
        distance_to_centroids is a function that returns an 
        Npts * K matrix where each row  of length k refers to 
        the square distance between the data point and the kth 
        centroid
        :return: Npts * K matrix
        """
        distance_matrix = np.zeros((len(X), self.K))
        for n in range(len(X)):
            for k in range(self.K):
                distance_matrix[n][k] = self.square_distance(X[n],
                                                             centroids[k])
        return distance_matrix

    def min_distance_cluster(self, distance_matrix):
        """
        min_distance_cluster is a function that returns 
        a vector assigning a cluster to each observation based on 
        the closest square distance.
        :return: vector of length Npts
        """
        return np.argmin(distance_matrix, 1)

    def distortion_measure(self, distance_matrix, labels):
        """
        distortion_measure is a function that computes the objective
        function we are trying to minimize.
        :return: scalar
        """
        return sum([distance_matrix[idx][i] for idx, i in enumerate(labels)])

    def check_convergence(self, it):
        """
        convergence is reached when there are no further changes
        on the cluster assignment. Note that the algorithm may 
        converge to a local rather than a global minimum.
        :return: Boolean
        """
        return self.J[it-1] - self.J[it] > self.eps

    def fit_kmeans(self, X):
        """
        fit_kmeans implements the kmeans algorithm:
        1st step: choose initial centroid values.
        First phase: E-step: 
        We minimize the objective function J with respect to the 
        min_distance_cluster vectors  keeping the centroids fix.
        Second phase: M-step: 
        We minimize J wrt the centroids keeping the indicator 
        vectors fixed.
        Convergence of the algorithm is assured.
        """
        it = 1
        self.J = [np.infty]
        iterate = True
        self.centroids = self.initial_centroids(X)
        while iterate:
            self.distance_matrix = self.distance_to_centroids(X, self.centroids)
            self.labels = self.min_distance_cluster(self.distance_matrix)
            self.J.append(self.distortion_measure(self.distance_matrix, self.labels))
            if self.check_convergence(it):
                self.centroids = self.compute_centroids(X, self.labels)
                it += 1
            else:
                iterate = False
        self.convergence_it = it