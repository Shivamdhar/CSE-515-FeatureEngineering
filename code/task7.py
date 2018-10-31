'''
This module is the program for task 7.
'''
from textual_descriptor_processor import TxtTermStructure
import sys
import math
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
import pickle

class Task7(object):
	def __init__(self):
		self.tensor_arr = []
		self.users = TxtTermStructure()
		self.images = TxtTermStructure()
		self.locations = TxtTermStructure()
		# get users term data
		self.users.load_users_data()
		# get images term data
		self.images.load_image_data()
		# get locations term data
		self.locations.load_location_data()

	'''
	Method: calculates tensor representation of the count of similar number
	of terms in between location-user-image pairs in the complete dataset
	'''
	def build_tensor(self):
		self.tensor_arr = []
		print("starting tensor build...")
		# iterating through locations to get common terms between locations-users-images
		for location in self.locations.master_dict:
			location_arr = []
			user_arr = []
			# set containing location-specific terms
			location_set = self.locations.get_terms(location)
			# iterating through all users for the given location
			for user in self.users.master_dict:
				user_arr = []
				# build set containing terms intersecting between user and location
				location_user_set = location_set.intersection(self.users.get_terms(user))
				# if no intersection, skip further check for images and mark all further
				# number of similar tags as zero for this user
				if(len(location_user_set) == 0):
					user_arr = list(np.zeros((len(self.images.master_dict.keys()),), dtype=int))
					location_arr.append(user_arr)
					continue
				# if there are some common terms between user and location, find further
				# common terms between all images and this set
				for image in self.images.master_dict:
					user_arr.append(len(location_user_set.intersection(self.images.get_terms(image))))
				# add list of count of common terms
				location_arr.append(user_arr)
				# print progress bar for tensor build
				sys.stdout.write('\r')
				sys.stdout.write("[%-30s] %d%%" % ('#'*len(self.tensor_arr), (100//len(self.locations.master_dict))*len(self.tensor_arr)))
				sys.stdout.flush()
			# add list of list of the common terms to parent list
			self.tensor_arr.append(location_arr)

	'''
	Method: given a tensor and value of k, computes the 
	CP-decomposition and returns the
	'''
	def compute_tensor_cp_decomposition(self, k):
		X = 0
		X = tl.tensor(self.tensor_arr)
		factors = parafac(X, rank=k)
		return factors

	'''
	Method: given a reduced vector space compute k
	groups of users/images/locations
	'''
	def compute_k_groups(self, k, factors, elem_keys):
		dict_sets_name = {0: 'location-group', 1:'user-group', 2:'image-group'}
		for i in range(0, len(factors)):
			k_groups = KMeans(n_clusters=k, random_state=0).fit_predict(factors[i])
			# print("k group length: ",k_groups.shape)
			# print(len(k_groups))
			similar_group_dict = dict()
			for j in range(0, len(k_groups)):
			    elem = k_groups[j]
			    # print(elem)
			    if elem in similar_group_dict:
			        # print('there')
			        similar_group_dict[elem].append(elem_keys[i][j])
			    else:
			        # print('not there')
			        # print(i,j)
			        similar_group_dict[elem] = [elem_keys[i][j]]
			print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
			print("-------------Grouping ",i," : ",dict_sets_name[i], "--------------------")
			print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
			for j in range(0, k):
			    print("Group ",j," : ", similar_group_dict[j])

	'''
	Method: runner implemented for all the tasks, takes user input, calculates tensor between 
	location users images using count for similar terms
	'''
	def runner(self):
		# take input from user
		k = int(input("Enter value of k: "))
		# build tensor for location-user-images based on count of common terms
		# self.build_tensor()
		print("\ncomputing CP tensor decomposition...")
		# factors = self.compute_tensor_cp_decomposition(k)
		elem_keys = []
		factors = []
		elem_keys.append(list(self.locations.master_dict.keys()))
		elem_keys.append(list(self.users.master_dict.keys()))
		elem_keys.append(list(self.images.master_dict.keys()))
		PIK1 = '../task7_k'+str(k)+'_loc.dat'
		loc_mat_decom = pickle.load(open(PIK1, 'rb'))
		PIK2 = '../task7_k'+str(k)+'_usr.dat'
		usr_mat_decom = pickle.load(open(PIK2, 'rb'))
		PIK3 = '../task7_k'+str(k)+'_img.dat'
		img_mat_decom = pickle.load(open(PIK3, 'rb'))
		# print(loc_mat_decom.shape)
		# print(usr_mat_decom.shape)
		# print(img_mat_decom.shape)
		factors.append(loc_mat_decom)
		factors.append(usr_mat_decom)
		factors.append(img_mat_decom)
		# print('len factors: ',np.array(factors).shape)
		self.compute_k_groups(k, np.array(factors), elem_keys)
