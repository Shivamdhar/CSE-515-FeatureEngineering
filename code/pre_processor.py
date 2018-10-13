'''
This module contains data preprocessing and data parsing methods.
'''
from collections import OrderedDict
import constants
from data_extractor import DataExtractor
import glob

class PreProcessor(object): 
# provide data object to be read
	def __init__(self):
		self.models = constants.MODELS
		data_extractor = DataExtractor()
		mapping = data_extractor.location_mapping()
		self.locations = list(mapping.values())

	def pre_process(self):
		self.remove_duplicates_from_visual_descriptor_dataset()
		self.rename_image_ids_from_visual_descriptor_dataset()

	def remove_duplicates_from_visual_descriptor_dataset(self):
		files = glob.glob(constants.VISUAL_DESCRIPTORS_DIR_PATH_REGEX)
		for file in files:
			raw_file_contents = open(file, "r").readlines()
			global_image_ids = []
			file_name = file.split("/")[-1]
			output_file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + file_name, "w")
			for row in raw_file_contents:
				image_id = row.split(",")[0]
				if image_id not in global_image_ids:
					output_file.write(row)

	def rename_image_ids_from_visual_descriptor_dataset(self):
		global_image_map = OrderedDict({})

		for location in self.locations:
			location_model_image_ids = []
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				location_model_image_ids += [row.split(",")[0] for row in data]
			global_image_map[location] = list(set(location_model_image_ids))

		location_files_to_be_cleaned = []
		dataset = list(global_image_map.values())

		for iterator1 in range(0, len(dataset)):
			for iterator2 in range(iterator1+1, len(dataset)):
				common_image_ids = set(dataset[iterator1]).intersection(dataset[iterator2])
				if len(common_image_ids) > 0:
					location_files_to_be_cleaned.append([self.locations[iterator1], self.locations[iterator2],
						                                 common_image_ids])

		for iterator in location_files_to_be_cleaned:
			for model in self.models:
				location_model_file = iterator[1] + " " + model + ".csv"
				data = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r").readlines()
				output_file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "w")
				for row in data:
					values = row.split(",")
					if values[0] in iterator[2]:
						image_id = values[0] + "_1"
						row = image_id + "," + ",".join(values[1:])
					output_file.write(row)


object = PreProcessor()
object.pre_process()