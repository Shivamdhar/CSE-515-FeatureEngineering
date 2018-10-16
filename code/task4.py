"""
This module is the program for task 4. 
"""
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from util import Util

class Task4(object):
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()

	def calculate_location_similarity(self, arr, location_list_indices, mapping, location_id):
		"""
		Method: calculate_location_similarity computes similarity score for the reduced location-location dataset.
		Given an input location, we need to find out similarity score of this location with respect to other locations.
		Computes similarity based on euclidean distance. For each comparison of an image in location 1 with all other
		images in location 2, we find out the most similar images. Finally, we return the average for these most similar
		images in location 2 with respect to location 1.
		Note that the low dimensional dataset will not have reference to visual descriptor models.
		k_semantics: low dimensional dataset to be used for similarity computation (total number of images X k)
		location_indices_map: stores key => location, value => indices in k_semantics
		algo_choice: (can be used in case we want to use a different similarity metric for each of the algorithms)
		input_location: reference location
		"""

		location_similarity = {}
		for location in location_list_indices.keys():
			imgximg_exhaustive_sim = []
			imgximg_similarity = []

			for i in range(0,location_list_indices[mapping[location_id]][1]):
				for j in range(location_list_indices[location][0],location_list_indices[location][1]):
					similarity = spatial.distance.euclidean(arr[i], arr[j])
					similarity = 1 / (1 + similarity)
					imgximg_exhaustive_sim.append(similarity)
				imgximg_similarity.append(max(imgximg_exhaustive_sim))
				imgximg_exhaustive_sim = []

			location_similarity.update({ location: sum(imgximg_similarity)/len(imgximg_similarity) })

		print(sorted(location_similarity.items(), key=lambda x: x[1], reverse=True)[:5])

	def runner(self):
		"""
		Method: runner implemented for all the tasks, takes user input, runs dimensionality reduction algorithm, prints
		latent semantics for input location and computes similarity between two locations for a given model using the
		latent semantics.
		"""

		#create the location_id-locationName mapping
		mapping = self.data_extractor.location_mapping()

		#take the input from user
		location_id = input("Enter the location id:")
		location = mapping[location_id]
		model = input("Enter the model: ")
		k = input("Enter value of k: ")
		algo_choice = input("Enter the Algorithm: ")

		#create the list of all files of the given model
		file_list = self.data_extractor.create_dataset(mapping, model, location_id)

		#append all the location images to a list with the first location being the input
		input_image_list, location_list_indices, input_location_index = self.data_extractor.append_givenloc_to_list(\
																			mapping, model,location_id, file_list)

		#convert list to numpy array
		input_image_arr = self.ut.convert_list_to_numpyarray(input_image_list)

		#select algorithm
		algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA, "LDA": self.ut.dim_reduce_LDA}

		#get the k latent semantics
		k_semantics = algorithms.get(algo_choice)(input_image_arr, k)

		print(k_semantics[0:input_location_index])

		self.calculate_location_similarity(k_semantics, location_list_indices, mapping, location_id)
