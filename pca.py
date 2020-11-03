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
        images_std = numerator/ ((images.std(axis=0)+eps).reshape(1, -1) )
        return images_std.T
    
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
        plt.title("Scree plot")
        plt.show()
    
    def sort_by_eigenv(self, eigenVal, eigenVect):
        idx = eigenVal.argsort()[::-1]   
        eigenVal = eigenVal[idx]
        eigenVect = eigenVect[:,idx]
        return eigenVal, eigenVect
    
    def compute_cov_matrix(self, data):
        #data must be centered
        N = data.shape[1]
        return np.dot(data, data.T)/(N-1)
        
    def spectral_decomp(self, cov_matrix):
        eigenValues, eigenVectors = linalg.eig(cov_matrix)
        return self.sort_by_eigenv(eigenValues, eigenVectors)
    
    def fit_pca(self, data):
        self.cov_matrix = self.compute_cov_matrix(data)
        self.eigenValues, self.eigenVectors = self.spectral_decomp(self.cov_matrix)
        self.new_coordinates = np.dot(self.eigenVectors[:,0: self.M].T,
                                      data)
        self.reconstruction = np.dot(self.eigenVectors[:,0: self.M], self.new_coordinates)
     
        