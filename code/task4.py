'''
This module is the program for task 4. 
'''
import numpy as np
from util import Util
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy import spatial

class Task_num_4(object):

	def calc_location_similarity(self, arr, location_list_indices, mapping, LocationID):
		location_similarity = {}
		for location in location_list_indices.keys():
			if(location == mapping[LocationID]):
				continue
			imgximg_sim = []
			for i in range(0,location_list_indices[mapping[LocationID]][1]):
				for j in range(location_list_indices[location][0],location_list_indices[location][1]):
					similarity = spatial.distance.euclidean(arr[i], arr[j])
					similarity = 1 / (1 + similarity)
					imgximg_sim.append(similarity)   
			location_similarity.update({location:sum(imgximg_sim)/len(imgximg_sim)})
		print(sorted(location_similarity.items(), key=lambda x: x[1], reverse=True)[:5])

	def task4(self):

		#create an instance of util class
		ut = Util()

		#create the locationID-locationName mapping
		mapping = ut.location_mapping()
		# print(mapping)

		#take the input from user
		LocationID = input("Enter the Location id:")
		Location = mapping[LocationID]
		print(Location)
		model = input("Enter the model: ")
		print(model)
		k = input("Enter value of k: ")
		print(k)
		algo_choice = input("Enter the Algorithim: ")
		print(algo_choice)

		#create the list of all files of the given model
		fileList = ut.create_dataset(mapping, model, LocationID)

		#append all the location images to a list with the first location being the input
		inputImageList, location_list_indices = ut.append_givenloc_to_list(mapping, model, LocationID, fileList)

		#convert list to numpy array
		inputImageArr = ut.convert_list_to_numpyarray(inputImageList)

		if(algo_choice == 'SVD'):
			svd = TruncatedSVD(n_components=int(k))
			svd.fit(inputImageArr)
			k_sematics_SVD = svd.transform(inputImageArr)
			print(k_sematics_SVD)
			self.calc_location_similarity(k_sematics_SVD, location_list_indices, mapping, LocationID)
		elif(algo_choice == 'PCA'):
			pca = PCA(n_components=int(k))
			k_semantics_PCA = pca.fit_transform(inputImageArr)
			print(k_semantics_PCA)
			self.calc_location_similarity(k_semantics_PCA, location_list_indices, mapping, LocationID)
		elif(algo_choice == 'LDA'):
			pass
