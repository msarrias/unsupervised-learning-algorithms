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
    
    def scree_plot_and_var_plot(self):
        fig = plt.figure(figsize=(15, 4))
        ax1 = plt.subplot(1, 2, 1)
        x = list(range(len(self.eigenValues)))
        y = self.eigenValues
        ax1.plot(x, y,c="blue")
        f = mticker.ScalarFormatter(useOffset=False,
                                    useMathText=True)
        g = lambda x, pos : "${}$".format(
            f._formatSciNotation('%1.10e' % x))
        plt.gca().yaxis.set_major_formatter(
            mticker.FuncFormatter(g))
        ax1.set_xlim([0,len(self.eigenValues)])
        ax1.set_ylim([0, max(self.eigenValues)])
        plt.xlabel ('i')
        plt.ylabel (r"$\lambda_{i}$")
        ax1.set_title("Scree plot")
        
        n_components = len(self.eigenValues)
        variance = sum(self.eigenValues)
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot([sum(self.eigenValues[0:i])/variance 
                  for i in range(n_components)],c="blue")
        ax2.set_xlim(0,n_components)
        ax2.set_ylim(0,1)
        plt.xlabel("Component")
        plt.ylabel("Variance Explained")
        plt.show()
        
    def explained_variance_plot(self):
        n_components = len(self.eigenValues)
        variance = sum(self.eigenValues)
        plt.plot([sum(self.eigenValues[0:i])/variance 
                  for i in range(n_components)],c="blue")
        plt.xlim(0,n_components)
        plt.ylim(0,1)
        plt.xlabel("Component")
        plt.ylabel("Variance Explained")
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
        self.reconstruction = np.dot(self.eigenVectors[:,0: self.M],
                                     self.new_coordinates)
     
        