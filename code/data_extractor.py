'''
This module contains data preprocessing and data parsing methods.
'''
from collections import OrderedDict
import constants
import numpy as np
from scipy import spatial
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

	'''
	Method: prepare_dataset_for_task5 takes locvation mapping and k as input to extract the required dataset i.e image
	feature data over all the models and locations.
	Returns - location_model_map : key => image id and value => features across all the models
	location_indices_map : key => location, value => indices of the corresponding location
	model_feature_length_map : key =>  model, value => length of feature set for each model
	'''
	def prepare_dataset_for_task5(self, mapping, k):
		locations = list(mapping.values())
		location_model_map = OrderedDict({})
		location_indices_map = OrderedDict({})
		model_feature_length_map = OrderedDict({})

		global_index_counter = 0

		for location in locations:
			for model in constants.MODELS:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				index_counter = 0

				for row in data:
					row_data = row.strip().split(",")
					feature_values = list(map(float, row_data[1:]))
					image_id = row_data[0]
					if image_id in location_model_map.keys():
						location_model_map[image_id] += feature_values
					else:
						location_model_map[image_id] = feature_values

					index_counter += 1
					if model not in model_feature_length_map.keys():
						model_feature_length_map[model] = len(feature_values)

				if location not in location_indices_map.keys():
					location_indices_map[location] = (global_index_counter, global_index_counter + index_counter)
					global_index_counter += index_counter

		return location_model_map, location_indices_map, model_feature_length_map

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