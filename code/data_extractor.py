"""
This module contains data preprocessing and data parsing methods.
"""
from collections import OrderedDict
import constants
import numpy as np
import os
from scipy import spatial
import glob
import os
import xml.etree.ElementTree as et

class DataExtractor(object): 
# provide data object to be read
	def location_mapping(self):
		#parse the xml file of the locations
		tree = et.parse(constants.DEVSET_TOPICS_DIR_PATH)
		#get the root tag of the xml file
		doc = tree.getroot()
		mapping = OrderedDict({})
		#map the location id(number) with the location name
		for topic in doc:
			mapping[topic.find("number").text] = topic.find("title").text

		return mapping

	def create_dataset(self, mapping, model, location_id):
		folder = constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
		location_names = list(mapping.values())
		file_list = []
		"""file_list contains list of tuples which are of the form [("location file path", "location") for all other
		locations other than the input location"""
		x = len(mapping)
		for i in range(0,x):
			if i != (int(location_id)-1):
				file_list.append((folder + location_names[i] + " " + model + ".csv", location_names[i]))

		return file_list

	''' Returns a map of location title to IDs '''
	def location_title_to_id_mapping(self):
		# Parse the xml file of the locations
		tree = et.parse("../dataset/text_descriptors/devset_topics.xml")
		
		# Get the root tag of the xml file
		doc = tree.getroot()
		mapping = OrderedDict({})
		
		# Map the location id(number) with the location name
		for topic in doc:
			mapping[topic.find('title').text] = (int)(topic.find('number').text)

		return mapping

	''' Gets all the files (paths) prefixed with the prefix given as parameter '''
	def get_all_files_prefixed_with(self, prefix):
		file_name_regex = '../dataset/visual_descriptors/processed/' + prefix + "*.csv"
		# All files of the given prefix (locationName)
		return [os.path.basename(x) for x in glob.glob(file_name_regex)]

	'''
	Method: prepare_dataset_for_task5 takes location mapping and k as input to extract the required dataset i.e image
	feature data over all the models and locations.
	Returns - location_model_map : key => image id and value => features across all the models
	location_indices_map : key => location, value => indices of the corresponding location
	model_feature_length_map : key =>  model, value => length of feature set for each model
	'''
	def prepare_dataset_for_task5(self, mapping, k):
		"""
		Method: prepare_dataset_for_task5 takes location mapping and k as input to extract the required dataset i.e image
		feature data over all the models and locations.
		Returns - location_model_map : key => image id and value => features across all the models
		location_indices_map : key => location, value => indices of the corresponding location
		model_feature_length_map : key =>  model, value => length of feature set for each model
		"""

		locations = list(mapping.values())
		location_model_map = OrderedDict({})
		location_indices_map = OrderedDict({})
		model_feature_length_map = OrderedDict({})

		global_index_counter = 0

		for location in locations:
			for model in constants.MODELS:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
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
		"""
		Method: append_givenloc_to_list takes the mapping, model, given location id, file_list as input to extract the required
		dataset i.e, feature data over a specific model and locations.
		Returns - input_image_list i.e, a list of all the images of all locations with the given location images at the start.
				- location_list_indices i.e, a dict of the start and end indices of all location"s images.
				- imput_location_index i.e, the end index of the given location
		"""

		folder = constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH
		location_list_indices = {}  
		input_image_list = []     
		given_file = folder + mapping[location_id] + " " + model + ".csv"

		with open (given_file) as f:
			given_file_data = (f.read()).split("\n")[:-1]

		for each_given_row in given_file_data:
			input_image_list.append(each_given_row.split(",")[1:])

		location_list_indices.update({ mapping[location_id]: [0, len(input_image_list)] })

		start = len(input_image_list)
		input_location_index = start;

		for each in file_list:
			each_file = each[0]
			title = each[1]

			with open (each_file) as e_file:
				file_data = (e_file.read()).split("\n")[:-1] #read data from each of the file in file_list

			for each_row in file_data:
				input_image_list.append(each_row.split(",")[1:])

			location_list_indices.update({ title:[start, len(input_image_list)] })
			start = len(input_image_list)

		return input_image_list, location_list_indices, input_location_index

	def prepare_dataset_for_task3(self, model, image_id):
		"""
		Method: Combining all the features of all locations for given color model.
				Assigning start index and end index to all locations in the combined matrix.
				Finding index of given input image.
		"""

		list_of_files = os.listdir(constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH)
		#dictionary of location name along with images and visual discriptor
		array_location_vector = {}
		start = 0
		image_input_array = []
		image_position = 0
		array_of_all_images = []

		for filename in list_of_files:
			if filename.endswith(model + ".csv"):
				loc = filename.replace(" " + model + ".csv","")
				#opening file with given model value
				with open(constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + filename,"r") as file:
					count = 0
					for index,line in enumerate(file):
						x = line.split(",")
						#Index of given input image
						if(x[0] == image_id):
							image_position = start + index
						#Array of all image IDs for all locations
						array_of_all_images.append(x[0])
						#Array of features of all images
						image_input_array.append(np.array(x[1:],dtype = np.float64))
						count += 1
					final = start + count - 1
					#Assigning start index and end index for all locations
					array_location_vector[loc] = [start,final]
					start = final + 1
		if not image_id in array_of_all_images:
			raise Exception("Wrong input: Image id not found")

		return array_of_all_images, image_input_array, image_position, array_location_vector
