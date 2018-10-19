"""
This module contains all functions used throughout the codebase. 
"""
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class Util(object):

	def __init__(self):
		self.data_extractor = DataExtractor()

	def get_similarity_scores(self,reference_vectors,input_vector):
		similarity_scores =	[]

		#Computing similarity between input vector and all the other vectors in reference_vectors matrix
		for vector in reference_vectors:
			result = spatial.distance.euclidean(input_vector,vector)
			result = 1 / (1 + result)
			similarity_scores.append(result)

		return similarity_scores

	def convert_list_to_numpyarray(self, data_list):
		numpy_arr = np.array(data_list, dtype=np.float64)

		return numpy_arr

	def dim_reduce_SVD(self, input_arr, k):
		svd = TruncatedSVD(n_components=int(k))
		svd.fit(input_arr)

		return(svd.transform(input_arr))

	def dim_reduce_PCA(self, input_arr, k):
		input_std = StandardScaler().fit_transform(input_arr)
		pca = PCA(n_components=int(k))

		return(pca.fit_transform(input_std))

	def dim_reduce_LDA(self, input_matrix, k):
		preprocessed_matrix = self.data_extractor.preprocessing_for_LDA(input_matrix)

		lda = LatentDirichletAllocation(n_components=int(k))

		return(lda.fit_transform(preprocessed_matrix))