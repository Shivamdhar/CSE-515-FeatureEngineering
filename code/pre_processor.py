"""
This module contains data preprocessing and data parsing methods.
This would be run before starting our driver program. It ensures that the raw dataset of visual descriptors is processed
and stored under processed directory.
This module contains one time tasks for pre-processing of data.
"""
from collections import OrderedDict
import constants
from data_extractor import DataExtractor
import glob
import pandas as pd
from textual_descriptor_processor import TxtTermStructure

class PreProcessor(object): 
	def __init__(self):
		self.models = constants.MODELS
		data_extractor = DataExtractor()
		mapping = data_extractor.location_mapping()
		self.locations = list(mapping.values())

	def pre_process(self):
		"""
		Any other preprocessing needed can be called from pre_process method.
		"""

		self.remove_duplicates_from_visual_descriptor_dataset()
		self.rename_image_ids_from_visual_descriptor_dataset()
		self.add_missing_objects_to_dataset()
		self.LDA_preprocessing()
		self.add_missing_words_to_term_spaces()

	def add_missing_words_to_term_spaces(self):
		global_words = {}
		
		txt_str = TxtTermStructure()
		txt_str.load_all_textual_data()

		users = TxtTermStructure()
		images = TxtTermStructure()
		locations = TxtTermStructure()
		users.load_users_data()
		images.load_image_data()
		locations.load_location_data()

		global_user_ids = list(users.master_dict.keys()) #all user ids
		global_user_words = set()

		for user_id in global_user_ids:
			global_user_words.update(users.get_terms(user_id))

		global_image_ids = list(images.master_dict.keys()) #all image ids
		global_image_words = set()
		
		for image_id in global_image_ids:
			global_image_words.update(images.get_terms(image_id))

		global_location_ids = list(locations.master_dict.keys()) #all location ids
		global_location_words = set()
		
		for location_id in global_location_ids:
			global_location_words.update(locations.get_terms(location_id))
		
		global_dict = txt_str.master_dict
		global_words = list(global_dict.keys())
		global_term_list = set()
		for key in global_words:
			global_term_list.update(txt_str.get_terms(key))
		
		words_not_in_location = global_term_list - global_location_words
		words_not_in_image = global_term_list - global_image_words
		words_not_in_user = global_term_list - global_user_words

		raw_file_contents = open(constants.TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerImage.txt", "r").readlines()
		write_image_file = open(constants.PROCESSED_TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerImage.txt", "w")
		count = 0
		for row in raw_file_contents:
			if count == 0:
				data = row.strip().split(" ")
				words_object_id = set()
				for i in range(0,len(data),4):
					words_object_id.update(data[i])
				not_present = words_not_in_image - words_object_id
				for word in not_present:
					data += [word, "0", "0", "0"]
				row = " ".join(data) + "\n"
				write_image_file.write(row)
			else:
				write_image_file.write(row)
			count += 1

		raw_file_contents = open(constants.TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerUser.txt", "r").readlines()
		write_user_file = open(constants.PROCESSED_TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerUser.txt", "w")
		count = 0
		for row in raw_file_contents:
			if count == 0:
				data = row.strip().split(" ")
				words_object_id = set()
				for i in range(0,len(data),4):
					words_object_id.update(data[i])
				not_present = words_not_in_user - words_object_id
				for word in not_present:
					data += [word, "0", "0", "0"]
				row = " ".join(data) + "\n"
				write_user_file.write(row)
			else:
				write_user_file.write(row)
			count += 1

		raw_file_contents = open(constants.TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerPOI.wFolderNames.txt", "r").readlines()
		write_location_file = open(constants.PROCESSED_TEXT_DESCRIPTORS_DIR_PATH + "devset_textTermsPerPOI.wFolderNames.txt", "w")
		count = 0
		for row in raw_file_contents:
			if count == 0:
				data = row.strip().split(" ")
				words_object_id = set()
				for i in range(0,len(data),4):
					words_object_id.update(data[i])
				not_present = words_not_in_location - words_object_id
				for word in not_present:
					data += [word, "0", "0", "0"]
				row = " ".join(data) + "\n"
				write_location_file.write(row)
			else:
				write_location_file.write(row)
			count += 1

	def remove_duplicates_from_visual_descriptor_dataset(self):
		"""
		Method: remove_duplicates_from_visual_descriptor_dataset removes any duplicate image id in a model file
		"""

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
		"""
		Method: rename_image_ids_from_visual_descriptor_dataset finds out if there are similar image ids across two
		locations and updates image id of one of the locations.
		FIX: 1) In case more than 1 location has same image ids with respect to a location, this method can"t handle such
		updates. (will need volunteer for fixing this :P)
		"""

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

	def add_missing_objects_to_dataset(self):
		"""
		Method: add_missing_objects_to_dataset finds out missing objects across locations and models and adds them to
		relevant files ensuring data consistency
		"""

		location_image_id_map = {}
		for location in self.locations:
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				file = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "r")
				data = file.readlines()
				file.close()
				location_model_image_ids = [row.split(",")[0] for row in data]
				for image_id in location_model_image_ids:
					if location in location_image_id_map.keys():
						if image_id not in location_image_id_map[location]:
							location_image_id_map[location].append(image_id)
					else:
						location_image_id_map[location] = [image_id]

		for location in self.locations:
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				data = pd.read_csv(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, dtype = \
						{0:'str'}, header=None)
				location_model_image_ids = data[0]
				missing_image_ids = set(location_image_id_map[location]) -\
									set(location_model_image_ids).intersection(set(location_image_id_map[location]))
				if len(missing_image_ids) > 0:
					file_des = open(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, "a")
					for image_id in missing_image_ids:
						feature_values = [str(data[iterator].mean()) for iterator in range(1, len(data.columns))]
						print("New object inserted : " + str(image_id) + " in " + location_model_file)
						file_des.write(str(image_id) + "," + ",".join(feature_values) + "\n")

	def LDA_preprocessing(self):
		"""
		Method: LDA_preprocessing scales the visual descriptor dataset to remove the negative values present.
		"""

		for location in self.locations:
			for model in self.models:
				location_model_file = location + " " + model + ".csv"
				data = pd.read_csv(constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, dtype = \
						{0:'str'}, header=None)
				length = data.shape[1]
				for i in range (1, length):
					column = data[i];
					min_val = column.min()
					if min_val < 0:
						for j in range(len(column)):
							column.iloc[j] -= min_val
				data.to_csv(constants.FINAL_PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH + location_model_file, sep = ',', \
							encoding = 'utf-8', index = False, header = False)

object = PreProcessor()
object.pre_process()