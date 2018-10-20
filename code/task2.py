"""
This module is the program for task 2.
"""
import constants
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from textual_descriptor_processor import TxtTermStructure
from util import Util

class Task2(object):
	"""
	This module is responsible for finding similarity between entities based on the latent semantics
	computed by PCA/LDA/SVD on original vector space (user/location/image)
	"""
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()


	def calculate_similarity(self, k_semantics_map, entity_id):
		'''
		Method : calculate_similarity computes compute the similarity 
		matrix for the given entity id and given the latent semantics vector
		in form of weights for given entity type
		'''
		k_semantics = [v for k,v in k_semantics_map]

		input_vector = k_semantics_map.get(entity_id,[])

		similarity_data = []

		if self.entity_type == 'location':
			mapping = self.data_extractor.location_mapping()

		for key,value in k_semantics_map.items():
			result = spatial.distance.euclidean(input_vector,value)
			result = 1 / (1 + result)
			if self.entity_type == 'location':
				k = mapping[k]
			elif self.entity_type == 'image':
				k = int(k)
			similarity_data.append((k,result))

		self.top_5(similarity_data)

	def top_5(self, similarity_data):
		"""
		Method: Prints the top5 similar entities (users/images/locations) with respect to input
		entity
		similarity_data: List of objects containing entity_id and the similarity score between
		respective entity
		"""
		print(sorted(similarity_data, key=lambda x: x[1], reverse=True)[:5])
		pass

	def runner(self):
		"""
		Method: runner implemented for all the tasks, 
		takes user input for type of entity from list of User,Image and Location
		and respective entity_id
		Displays the top 5 entities with respect to input entity 
		using the latent semantics obtained from task1 for respective entity vector space
		"""
		#k = input("Enter the value of k :")
		entity_index = int(input("Choose the entity id \t1) User \t2)Image \t3)Location.: "))
		if entity_index == 1:
			self.entity_type = constants.USER_TEXT
			user_id = input("Enter the user id: ")
			'''get the users semantics map with 
			k_semantics = get_k_semantics(users)
			'''
			k_semantics_map = {}
			if user_id not in k_semantics_map:
				raise KeyError(constants.USER_ID_KEY_ERROR)
			self.calculate_similarity(k_semantics_map,user_id)
		elif entity_index == 2:
			self.entity_type = constants.IMAGE_TEXT
			image_id = input("Enter the image id: ")
			'''get the images semantics
			k_semantics_map = get_k_semantics_map("images")
			'''
			k_semantics_map = {}
			if image_id not in k_semantics_map:
				raise KeyError(constants.IMAGE_ID_KEY_ERROR)
			self.calculate_similarity(k_semantics_map,image_id)
		elif entity_index == 3:
			self.entity_type = constants.LOCATION_TEXT
			location_id = input("Enter the location id: ")
			'''get the locations semantics
			k_semantics_map = get_k_semantics_map("locations")
			'''
			k_semantics_map = {}
			if location_id not in k_semantics_map:
				raise KeyError(constants.LOCATION_ID_KEY_ERROR)
			self.calculate_similarity(k_semantics_map,location_id)
