'''
This module is the program for task 5.
'''
from collections import OrderedDict
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from util import Util

class Task5(object):
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()

	'''
	Method: calculate_location_similarity computes similarity score for the reduced location-location dataset.
	Given an input location, we need to find out similarity score of this location with respect to other locations.
	Note that the low dimensional dataset will not have reference to visual descriptor models.
	k_semantics: low dimensional dataset to be used for similarity computation (total number of images X k)
	location_indices_map: stores key => location, value => indices in k_semantics
	algo_choice: (can be used in case we want to use a different similarity metric for each of the algorithms)
	input_location: reference location
	'''
	def calculate_location_similarity(self, k_semantics, location_indices_map, algo_choice, input_location):
		# TODO: If we want to use different similarity metrics for the three models, following switcher can be used.
		# similarity_computation = { "SVD": self.distance_based_similarity_computation,
		# 							"PCA": self.distance_based_similarity_computation }

		locations = list(self.mapping.values())
		location_location_similarity_map = OrderedDict({})
		location1_indices = location_indices_map[input_location]
		location1_data = k_semantics[location1_indices[0]:location1_indices[1]]

		for location_index2 in range(0,len(locations)):
				location2_indices = location_indices_map[locations[location_index2]]
				location2_data = k_semantics[location2_indices[0]:location2_indices[1]]
				# similarity_score = similarity_computation.get(algo_choice)(location1_data, location2_data)
				similarity_score = self.distance_based_similarity_computation(location1_data, location2_data)
				location_location_similarity_map[locations[location_index2]] = similarity_score

		self.top_5(location_location_similarity_map)

	'''
	Method: distance_based_similarity_computation computes similarity based on euclidean distance.
	For each comparison of an image in location1_data with all other images in location2_data, we find out the most
	similar images. Finally, we return the average for these most similar images in location2_data with respect to
	location1_data.
	location1_data: Low dimensional dataset for input location (number of images in location 1 X k)
	location2_data: Low dimensional dataset for other locations (number of images in location 2 X k)
	'''
	def distance_based_similarity_computation(self, location1_data, location2_data):
		image_image_similarity = []
		for iterator1 in location1_data:
			local_img_img_similarity = []
			for iterator2 in location2_data:
				distance = spatial.distance.euclidean(iterator1, iterator2)
				similarity = 1 / (1 + distance)
				local_img_img_similarity.append(similarity)
			image_image_similarity.append(max(local_img_img_similarity))

		return sum(image_image_similarity)/len(image_image_similarity)

	'''
	Method: top_5 prints the top 5 most similar locations with respect to the input location.
	location_location_similarity_map: stores similarity score between input and other locations in dataset.
	'''
	def top_5(self, location_location_similarity_map):
		print(sorted(location_location_similarity_map.items(), key=lambda x: x[1], reverse=True)[:5])

	'''
	Method: print_latent_semantics_for_input_location prints k latent semantics for the location input by user
	k_semantics: low dimensional location-location dataset (total number of images X k)
	input_location: user input
	location_indices_map:  stores key => location, value => indices in k_semantics
	'''
	def print_latent_semantics_for_input_location(self, k_semantics, input_location, location_indices_map):
		location_indices = location_indices_map[input_location]
		print(k_semantics[location_indices[0]:location_indices[1]])

	'''
	Method: runner implemented for all the tasks, takes user input, runs dimensionality reduction algorithm, prints
	latent semantics for input location and computes similarity between two locations using the latent semantics.
	'''
	def runner(self):
		#take input from user
		location_id = input("Enter the location id:")
		input_location = self.mapping[location_id]
		k = input("Enter value of k: ")
		algo_choice = input("Enter the Algorithm: ")

		data, location_indices_map, model_feature_length_map = self.data_extractor.prepare_dataset_for_task5\
																					(self.mapping, k)
		# model_feature_length_map is unused but if any code change is required, this will be handy so will retain this.

		matrix = np.array(list(data.values()))
		algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA, "LDA": self.ut.dim_reduce_LDA }

		k_semantics = algorithms.get(algo_choice)(matrix, k)

		self.print_latent_semantics_for_input_location(k_semantics, input_location, location_indices_map)
		self.calculate_location_similarity(k_semantics, location_indices_map, algo_choice, input_location)