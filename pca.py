import numpy as np
import numpy.linalg as linalg
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

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
    
    def plot_eigenspectrum(self):
        x = list(range(len(self.eigenValues)))
        y = self.eigenValues
        plt.plot(x, y,c="blue")
        f = mticker.ScalarFormatter(useOffset=False,
                                    useMathText=True)
        g = lambda x, pos : "${}$".format(
            f._formatSciNotation('%1.10e' % x))
        plt.gca().yaxis.set_major_formatter(
            mticker.FuncFormatter(g))
        plt.xlim([0,len(self.eigenValues)])
        plt.ylim([0, max(self.eigenValues)])
        plt.xlabel ('i')
        plt.ylabel (r"$\lambda_{i}$")
        plt.title("Eigenvalue spectrum")
        plt.show()

    
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
        self.cov_matrix = self.compute_covariance_matrix(standardized_images)
        self.eigenValues, self.eigenVectors = self.spectral_decomposition(self.cov_matrix)
        self.new_coordinates = np.dot(self.eigenVectors[:,0: self.M].T,
                                      standardized_images.T)       
        
        