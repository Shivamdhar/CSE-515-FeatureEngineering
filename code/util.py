'''
This module contains all functions used throughout the codebase. 
'''
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

class Util(object):

	def convert_list_to_numpyarray(self, data_list):
		numpy_arr = np.array(data_list, dtype=np.float64)

		return numpy_arr

	def dim_reduce_SVD(self, input_image_arr, k):
		svd = TruncatedSVD(n_components=int(k))
		svd.fit(input_image_arr)

		return(svd.transform(input_image_arr))

	def dim_reduce_PCA(self, input_image_arr, k):
		pca = PCA(n_components=int(k))

		return(pca.fit_transform(input_image_arr))


		
		
