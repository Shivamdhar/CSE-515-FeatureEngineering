'''
This module contains data preprocessing and data parsing methods.
'''
from collections import OrderedDict
import constants
import numpy as np
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
		mapping = OrderedDict({})
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

	def prepare_dataset_for_task5(self, mapping, k):
		locations = list(mapping.values())
		location_model_map = OrderedDict({})
		for location in locations:
			for model in constants.MODELS:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				for row in data:
					row_data = row.strip().split(",")
					feature_values = list(map(float, row_data[1:]))
					image_id = row_data[0]
					if image_id in location_model_map.keys():
						location_model_map[image_id] += feature_values
					else:
						location_model_map[image_id] = feature_values


		return location_model_map

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