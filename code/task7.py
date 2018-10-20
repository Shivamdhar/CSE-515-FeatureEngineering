'''
This module is the program for task 7.
'''
from textual_descriptor_processor import TxtTermStructure

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

class Task7(object):
	def __init__(self):
		self.tensor_arr = []

	'''
	Method: calculates tensor representation of the count of similar number
	of terms in between location-user-image pairs in the complete dataset
	'''
	def build_tensor(self):
		# get users term data
	    users = TxtTermStructure()
	    users.loadUsersData()
	    # get images term data
	    images = TxtTermStructure()
	    images.loadImageData()
	    # get locations term data
	    locations = TxtTermStructure()
	    locations.loadLocationData()
	    
	    self.tensor_arr = []
	    print("starting tensor build...")
	    # iterating through locations to get common terms between locations-users-images
	    for location in locations.master_dict:
	        location_arr = []
	        user_arr = []
	        # set containing location-specific terms
	        location_set = locations.getTerms(location)
	        # iterating through all users for the given location
	        for user in users.master_dict:
	            user_arr = []
	            # build set containing terms intersecting between user and location
	            location_user_set = location_set.intersection(users.getTerms(user))
	            # if no intersection, skip further check for images and mark all further
	            # number of similar tags as zero for this user
	            if(len(location_user_set) == 0):
	                user_arr = list(np.zeros((len(images.master_dict.keys()),), dtype=int))
	                location_arr.append(user_arr)
	                continue
	            # if there are some common terms between user and location, find further
	            # common terms between all images and this set
	            for image in images.master_dict:
	                user_arr.append(len(location_user_set.intersection(images.getTerms(image))))
	            # add list of count of common terms
	            location_arr.append(user_arr)
	        # add list of list of the common terms to parent list
	        self.tensor_arr.append(location_arr)

	'''
	Method: given a tensor and value of k, computes the 
	CP-decomposition and returns the
	'''
	def compute_tensor_cp_decomposition(self, k):
		X = 0
		X = tl.tensor(self.tensor_arr)
		factors, wt, err = parafac(X, rank=k)
		print(factors)


	'''
	Method: runner implemented for all the tasks, takes user input, calculates tensor between 
	location users images using count for similar terms
	'''
	def runner(self):
		# take input from user
		k = input("Enter value of k: ")
		# build tensor for location-user-images based on count of common terms
		self.build_tensor()
		self.compute_tensor_cp_decomposition(k)
