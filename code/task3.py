'''
This module is the program for task 3. 
'''
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from util import Util

class Task_3(object):
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()


	''' image-image and image-location '''
	def calculate_similarity(self, k_semantics, image_position, array_of_all_images, array_location_vector):
		vector_of_input_image = k_semantics[image_position]
		similarity_score_images = []

		for vector in k_semantics:
			result = spatial.distance.euclidean(vector_of_input_image,vector)
			result = 1/(1+result)
			similarity_score_images.append(result)

		image_and_score=[]
		for i in range(len(array_of_all_images)):
			image_and_score.append([array_of_all_images[i],similarity_score_images[i]])

		sorted_sim_vector = sorted(image_and_score,key=lambda x:x[1],reverse=True) #sorting the similarity vector
		print("5 most similary images with matching score is :")
		print(sorted_sim_vector[:5])

		loc_img_score=[]
		temp=[]
		for key in array_location_vector:
			start_index = array_location_vector[key][0]
			end_index = array_location_vector[key][1]
			temp=sorted(similarity_score_images[start_index:end_index+1],key=lambda x:x,reverse=True)[0]
			mapping = self.data_extractor.location_mapping()
			for loc_id,location_name in mapping.items():
				if(key == location_name):
					location_id = loc_id
			loc_img_score.append([location_id,key,temp])

		temp_1=sorted(loc_img_score,key=lambda x:x[2],reverse=True)[:5]
		print("5 most similary locations with matching score is :")
		print(temp_1)

	def runner(self):
		model = input("Enter the model : ")
		k = input("Enter the value of k :")
		image_ID = input("Enter image ID : ")

		array_of_all_images, image_input_array, image_position, \
		array_location_vector = self.data_extractor.prepare_dataset_for_task3(model, image_ID)

		algo_choice = input("Enter the Algorithim: ")

		algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA , "LDA": self.ut.dim_reduce_LDA}

		k_semantics = algorithms.get(algo_choice)(image_input_array, k)
		print(k_semantics)

		self.calculate_similarity(k_semantics, image_position, array_of_all_images, array_location_vector)