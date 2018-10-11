'''
This module is the program for task 4. 
'''
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from util import Util

class Task_num_4(object):

	def calculate_location_similarity(self, arr, location_list_indices, mapping, location_id):
		location_similarity = {}
		for location in location_list_indices.keys():
			if(location == mapping[location_id]):
				continue

			imgximg_sim = []
			for i in range(0,location_list_indices[mapping[location_id]][1]):
				for j in range(location_list_indices[location][0],location_list_indices[location][1]):
					similarity = spatial.distance.euclidean(arr[i], arr[j])
					similarity = 1 / (1 + similarity)
					imgximg_sim.append(similarity)   
			location_similarity.update({ location: sum(imgximg_sim)/len(imgximg_sim) })

		print(sorted(location_similarity.items(), key=lambda x: x[1], reverse=True)[:5])

	def task4(self):
		#create an instance of util class
		ut = Util()
		data_extractor = DataExtractor()

		#create the location_id-locationName mapping
		mapping = data_extractor.location_mapping()

		#take the input from user
		location_id = input("Enter the location id:")
		location = mapping[location_id]
		model = input("Enter the model: ")
		k = input("Enter value of k: ")
		algo_choice = input("Enter the Algorithim: ")

		#create the list of all files of the given model
		file_list = data_extractor.create_dataset(mapping, model, location_id)

		#append all the location images to a list with the first location being the input
		input_image_list, location_list_indices = data_extractor.append_givenloc_to_list(mapping, model, location_id, file_list)

		#convert list to numpy array
		input_image_arr = ut.convert_list_to_numpyarray(input_image_list)

		k_semantics = ""

		if(algo_choice == 'SVD'):
			k_semantics = ut.dim_reduce_SVD(input_image_arr, k)
			print(k_semantics)
		elif(algo_choice == 'PCA'):
			k_semantics = ut.dim_reduce_PCA(input_image_arr, k)
			print(k_semantics)
		elif(algo_choice == 'LDA'):
			pass

		self.calculate_location_similarity(k_semantics, location_list_indices, mapping, location_id)

