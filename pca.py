import numpy as np
import numpy.linalg as linalg

class pca:   
    def __init__(self, M):
        self.M = M

    @staticmethod
    def standardize_images_channel_wise(images, eps):
        '''
        Channel-wise normalization of the input images: 
        subtracted by mean and divided by std along the pixels
        '''
        N, H, W = images.shape
        images = np.reshape(images, (-1, H * W))
        numerator = images - images.mean(axis=0).reshape(1, -1)
        images_standardized = numerator/ ((images.std(axis=0)+eps).reshape(1, -1) )
        return images_standardized
    
    def sort_by_eigenv(self, eigenVal, eigenVect):
        idx = eigenVal.argsort()[::-1]   
        eigenVal = eigenVal[idx]
        eigenVect = eigenVect[:,idx]
        return eigenVal, eigenVect
    
    def compute_covariance_matrix(self, standardized_images):
        N = len(standardized_images)
        return np.dot(standardized_images.T, standardized_images)/N
        
    def spectral_decomposition(self, cov_matrix):
        eigenValues, eigenVectors = linalg.eig(cov_matrix)
        return self.sort_by_eigenv(eigenValues, eigenVectors)
    
    def fit_pca(self, standardized_images):
        cov_matrix = self.compute_covariance_matrix(standardized_images)
        self.eigenValues, self.eigenVectors = self.spectral_decomposition(cov_matrix)
        self.new_coordinates = np.dot(self.eigenVectors[:,0: self.M].T,
                                      standardized_images.T)       
        
        