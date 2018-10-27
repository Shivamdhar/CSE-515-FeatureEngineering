import constants
from data_extractor import DataExtractor
from datetime import datetime
import numpy
from operator import itemgetter
import pandas as pd
from util import Util


class Task6:
	
	def __init__(self):
		"""
		Method Explanation:
			Intializes all the variables for the analysis task.
		"""
		self.util = Util()
		self.data_extractor = DataExtractor()

		self.location_id_to_title_map = self.data_extractor.location_mapping()
		self.location_title_to_id_map = self.data_extractor.location_title_to_id_mapping()
	
		self.location_list = list(self.location_title_to_id_map.values()) # List of location ids
		self.LOCATION_COUNT = len(self.location_list) # constant

		self.global_term_dictionary_current_index = 0 # To store the count of unique terms and indexing a given term in the global dictionary
		self.global_term_dictionary = dict() # To store the global list of terms as keys and their indices as values
		self.global_term_index_dictionary = dict() # To store the global list of terms referenced via the indices as the keys and terms as the values
		self.location_dictionary = dict() # To store the terms of a particular location and their corresponding attributes
		self.similarity_matrix = numpy.zeros((self.LOCATION_COUNT, self.LOCATION_COUNT)) # To capture location-location similarity

	def construct_vocabulary(self):
		"""
		Method Explanation:
			. Constructs a global term vocabulary.
			. Constructs a location based term vocabulary.
		"""
		with open(constants.TEXTUAL_DESCRIPTORS_DIR_PATH + "devset_textTermsPerPOI.txt", encoding="utf-8") as f:
			lines = [line.rstrip("\n") for line in f]
			for line in lines:
				words = line.split()

				temp_list_for_title = []
				# extract location title
				while "\"" not in words[0]:
					temp_list_for_title.append(words.pop(0))
				location_title = ("_").join(temp_list_for_title)
				location_id = self.location_title_to_id_map[location_title]

				# Build the term vocabulary and also the dictionary for terms corresponding to the locations and their scores
				for index, word in enumerate(words):
					index_mod4 = index%4
					
					if index_mod4 == 0: # the term
						current_word = word.strip('\"')
						if not self.global_term_dictionary.get(current_word):
							self.global_term_dictionary[current_word] = self.global_term_dictionary_current_index
							self.global_term_index_dictionary[self.global_term_dictionary_current_index] = current_word
							self.global_term_dictionary_current_index+= 1
						if not self.location_dictionary.get(location_id):
							self.location_dictionary[location_id] = {}
						if not self.location_dictionary.get(location_id).get(current_word):
							self.location_dictionary[location_id][current_word] = { "TF": 0, "DF": 0, "TFIDF": 0}
					elif index_mod4 == 1: # TF
						self.location_dictionary[location_id][current_word]["TF"] = int(word)
					elif index_mod4 == 2: # DF
						self.location_dictionary[location_id][current_word]["DF"] = int(word)
					elif index_mod4 == 3: # TFIDF
						self.location_dictionary[location_id][current_word]["TFIDF"] = float(word)

	def construct_similarity_matrix(self, model):
		"""
		Method Explanation:
			. Goes over every location as a potential query location, compares its textual descriptors with every other location as a
			  potential target location.
			. The comparison is based on the Cosine Similarity scores of one of the model vectors (TF/DF/TFIDF) defined by the <model> parameter.
		Inputs:
			<model> - Has three possible values -- TF, DF, TFIDF. Corresponds to which model score to consider for computing the Cosine Similarity
			          between the textual descriptors.
		"""
		the_model = model
		# Go over every location as a potential query location
		for query_location_id in self.location_list:
			query_model_vector = [0] * self.global_term_dictionary_current_index
			
			# Construct the query model vector (<the_model> values of each term in the query location)
			for current_term_id_key, current_term_id_value in self.location_dictionary[query_location_id].items():
				if current_term_id_key == the_model:
					continue
				current_term_index = self.global_term_dictionary[current_term_id_key]
				query_model_vector[current_term_index] = self.location_dictionary[query_location_id][current_term_id_key][the_model] 

			# Go over every location as a potential target location
			for target_location_id, target_location_id_data in self.location_dictionary.items():
				# If query location is the same as target location, similarity = 1
				if target_location_id == query_location_id:
					self.similarity_matrix[query_location_id - 1][target_location_id - 1] = 1
					continue
				else:
					if not self.location_dictionary.get(target_location_id).get(the_model):
						self.location_dictionary[target_location_id][the_model] = [0] * self.global_term_dictionary_current_index
	
					# Build the target model vector comprising of the_model scores of the target location
					for current_term_key, current_term_value in self.location_dictionary[target_location_id].items():
						if current_term_key == the_model:
							continue
						current_term_index = self.global_term_dictionary[current_term_key]
						self.location_dictionary[target_location_id][the_model][current_term_index] = self.location_dictionary[target_location_id][current_term_key][the_model]
					
					# Compute the Cosine Similarity between the query model vector and target model vector
					cosine_similarity_value = self.util.cosine_similarity(query_model_vector, self.location_dictionary[target_location_id][the_model])
					self.similarity_matrix[query_location_id - 1][target_location_id - 1] = cosine_similarity_value
	
	def print_k_latent_semantics(self, k):
		"""
		Method Explanation:
			. Applies a Singular Valued Decomposition on the similarity matrix and prints the first k latent semantics determined by the k parameter.
			. The output is in the form of location-weight pairs for each semantic sorted in the decreasing order of weights.
		Input:
			. <k> for considering only the k latent semantics post SVD
		"""
		U, S, Vt = numpy.linalg.svd(self.similarity_matrix)

		# Get the concept mapping
		concept_mapping = self.similarity_matrix.dot(U[:,:k])
		concept_mapping = concept_mapping.transpose()

		# {
		#  <location_id>: [{"Location Name": <>, "Weight": <>}, {"Location Name": <>, "Weight": <>}, ...],
		#  <location_id>: [{"Location Name": <>, "Weight": <>}, {"Location Name": <>, "Weight": <>}, ...],
		#  ...
		# }
		semantic_data_dict = {}
		print("")
		for arr_index, arr in enumerate(concept_mapping):
			current_key = arr_index+1
			if not semantic_data_dict.get(current_key):
				semantic_data_dict[current_key] = []

			for index, element in enumerate(arr):
				semantic_data_dict[current_key].append({ "Location Name": self.location_id_to_title_map[str(index+1)], "Weight": element })

			# Sort the latent semantic based on the weight of the feature
			sorted_list = sorted(semantic_data_dict[current_key], key=itemgetter("Weight"), reverse=True) 
			semantic_data_dict[current_key].clear()
			semantic_data_dict[current_key] = sorted_list
			
			# Print location name-weight pairs sorted in decreasing order of weights
			print("Latent Semantic: ", current_key)
			for idx, data in enumerate(sorted_list):
				print("\tLocation Name: ", semantic_data_dict[current_key][idx]["Location Name"], " | Weight: ", semantic_data_dict[current_key][idx]["Weight"])
			print("")

	def runner(self):
		k = input("Enter the k value: ")
		k = int(k)
		the_model = "TFIDF"
		self.construct_vocabulary()
		self.construct_similarity_matrix(the_model)
		self.print_k_latent_semantics(k)
