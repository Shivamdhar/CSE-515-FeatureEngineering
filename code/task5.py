'''
This module is the program for task 4. 
'''
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from util import Util

class Task5(object):
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()

	def calculate_location_similarity(self):
		#TODO: Add computation for location-location similarity
		pass

	def runner(self):
		#create the location_id-locationName mapping
		mapping = self.data_extractor.location_mapping()

		#take the input from user
		location_id = input("Enter the location id:")
		location = mapping[location_id]
		k = input("Enter value of k: ")
		algo_choice = input("Enter the Algorithim: ")

		data = self.data_extractor.prepare_dataset_for_task5(mapping, k)
		matrix = np.array(list(data.values()))

		algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA }

		k_semantics = algorithms.get(algo_choice)(matrix, k)

		print(k_semantics[0:3])