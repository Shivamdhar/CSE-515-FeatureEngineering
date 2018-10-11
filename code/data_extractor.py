'''
This module contains data preprocessing and data parsing methods.
'''
import numpy as np
import os 
from scipy.spatial.distance import cosine as cs
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import xml.etree.ElementTree as et

class DataExtractor(object): 
# provide data object to be read
	def location_mapping(self):
		#parse the xml file of the locations
		tree = et.parse("../dataset/text_descriptors/devset_topics.xml")
		#get the root tag of the xml file
		doc = tree.getroot()
		mapping = {}
		#map the location id(number) with the location name
		for topic in doc:
			mapping[topic.find('number').text] = topic.find('title').text

		return mapping

	def create_dataset(self, mapping, model, location_id):
		folder = "../dataset/visual_descriptors/"
		location_names = list(mapping.values())
		file_list = []
		'''file_list contains list of tuples which are of the form [('location file path', 'location') for all other 
		locations other than the input location'''
		x = len(mapping)
		for i in range(0,x):
			if i != (int(location_id)-1):
				file_list.append((folder + location_names[i] + " " + model + ".csv", location_names[i]))

		return file_list

	def append_givenloc_to_list(self, mapping, model, location_id, file_list):
		folder = "../dataset/visual_descriptors/"	
		location_list_indices = {}  
		input_image_list = []     
		given_file = folder + mapping[location_id] + " " + model + ".csv"

		with open (given_file) as f:
			given_file_data = (f.read()).split("\n")[:-1]

		for each_given_row in given_file_data:
			input_image_list.append(each_given_row.split(",")[1:])

		location_list_indices.update({ mapping[location_id]: [0, len(input_image_list)] })

		start = len(input_image_list)
		for each in file_list:
			each_file = each[0]
			title = each[1]

			with open (each_file) as e_file:
				file_data = (e_file.read()).split("\n")[:-1]#read data from each of the file in file_list

			for each_row in file_data:
				input_image_list.append(each_row.split(",")[1:])

			location_list_indices.update({ title:[start, len(input_image_list)] })
			start = len(input_image_list)

		return input_image_list, location_list_indices