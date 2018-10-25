import numpy
import pandas as pd
from sklearn import preprocessing
from operator import itemgetter
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from util import Util
import constants
from data_extractor import DataExtractor
import numpy

class Task6:

	'''
	Method Explanation:
		Given two dataframes of image data, the function
		. Applies Min Max normalization.
		. Computes euclidean distance between each pair of reference and current dataframe images.
		. Sums the distances up and returns the average from the sorted list of euclidean distances as an estimated measure of the distance between the two data frames.		
	Inputs:
		. reference_df_values - the data frame values corresponding to the image data of the current model of the query location.
		. current_df_values - the data frame vaues corresponding to the image data of the corresponding model of the current target location.
		. util_reference - a reference to the util object instantiated in the runner for invoking the euclidean distance computation helper defined elsewhere.
	Output:
		A distance value (float) between the reference_df_values and the current_df_values derived by the explanation above.
	'''
	def get_the_euclidean_distance_value(self, reference_df_values, current_df_values, util_reference):

		distance_list = []

		# Apply MinMax Normalization
		min_max_scaler = preprocessing.MinMaxScaler()
		reference_df_scaled = min_max_scaler.fit_transform(reference_df_values)
		current_df_scaled = min_max_scaler.fit_transform(current_df_values)
		normalized_reference_df_values = pd.DataFrame(reference_df_scaled)
		normalized_current_df_values = pd.DataFrame(current_df_scaled)

		# Compute the distance
		normalized_reference_df_values = normalize(reference_df_values, norm='l2')
		normalized_current_df_values = normalize(current_df_values, norm='l2')
		for reference_data_object_index, reference_data_object_value in enumerate(normalized_reference_df_values):
			for current_data_object_index, current_data_object_value in enumerate(normalized_current_df_values):
				distance_list.append(util_reference.compute_euclidean_distance(reference_data_object_value, current_data_object_value))
		return (sum(distance_list)/len(distance_list))
	
	'''
	Method Explanation:
		Orchestrator to enable (if needed) different distance metrics for different model representations of the images.
	Inputs:
		. reference_df_values - the data frame values corresponding to the image data of the current model of the query location.
		. current_df_values - the data frame vaues corresponding to the image data of the corresponding model of the current target location.
		. reference_sample_count - to enable consideration of distance computation based on the reference sample count alone (if necessary)
		. model_name - to enable differentiation between distance metrics based on the model that is chosen.
		. util_reference - a reference to the util object instantiated in the runner for invoking the euclidean distance computation helper defined elsewhere.
	Output:
		A distance value between the reference_df_values and current_df_values derived based on the model_name that is provided.
	'''
	def get_the_distance_value(self, reference_df_values, current_df_values, reference_sample_count, model_name, util_reference):
		# If we have different distance/similarity metrics, we can split based on model_name here.
		return self.get_the_euclidean_distance_value(reference_df_values, current_df_values, util_reference)

	def runner(self):
		k = input('Enter the k value: ')
		k = int(k)

		util = Util()
		data_extractor = DataExtractor()

		location_id_to_title_map = data_extractor.location_mapping()
		location_title_to_id_map = data_extractor.location_title_to_id_mapping()

		location_list = list(location_id_to_title_map.values())
	
		LOCATION_COUNT = len(location_list) # constant
		MODEL_COUNT = len(constants.MODELS)
		MAX_SCORE = (LOCATION_COUNT-1)*MODEL_COUNT
		
		FILE_PATH_PREFIX = constants.PROCESSED_VISUAL_DESCRIPTORS_DIR_PATH # '../dataset/visual_descriptors/processed/' # constant
		# {
		# 	1: {'CM': [{'location_id': 1, 'distance': 0}, {'location_id':2, 'distance': 0.45}, ...], 'CN': [...], ... },
		# 	2: {'CM': [...], 'CN': [...], ...},
		#   ... ,
		#   <query_location>: {
		# 						<model>: [{'location_id': <location_id>, 'distance': <distance>}, {'location_id': <location_id>, 'distance': <distance>}],
		# 					   	<model>: [...],
		# 					   	...
		# 					  }
		# }
		global_location_distance_data_dict = {}
		# {
		# 1: {1: 0, 2: 0.54, 3: 0.43, ...},
		# 2: { 1: 0.45, 2: 0, ...},
		# ... ,
		# <query_location>: { <target_location>: <distance>, <target_location>: <distance>, ...}
		# }
		location_wise_distance_data_dict = {}
		similarity_matrix = numpy.zeros((LOCATION_COUNT, LOCATION_COUNT))
		print('Starting...')

		# Go over every location as a potential query location
		for query_location in location_list:
			query_location_files = data_extractor.get_all_files_prefixed_with(query_location)
			query_location_id = location_title_to_id_map[query_location]

			if not global_location_distance_data_dict.get(query_location_id):
				global_location_distance_data_dict[query_location_id] = {}
			if not location_wise_distance_data_dict.get(query_location_id):
				location_wise_distance_data_dict[query_location_id] = {}
			print('Query Location: ', query_location)

			# Go over every model file in the query location
			for query_model_file in query_location_files:
				query_model_name_with_csv = query_model_file.split(" ")[1] # CM.csv, CN.csv, <modelName>.csv, ...
				query_model = query_model_name_with_csv.split(".")[0] # CM, CN, CN3x3, <modelName>, ...
				query_file_path = FILE_PATH_PREFIX + query_model_file
				query_model_df = pd.read_csv(query_file_path, header=None)
				del query_model_df[0]
				query_model_df = query_model_df.reset_index(drop=True)
				query_model_df_row_count = query_model_df.shape[0]

				if not global_location_distance_data_dict.get(query_location_id).get(query_model):
					global_location_distance_data_dict[query_location_id][query_model] = []
				print('\tQuery Model: ', query_model)

				# Go over every location as a potential target location for which we will compute the distance to from the query location
				for target_location in location_list:
					target_location_id = location_title_to_id_map[target_location]
					# If query location == target location, distance = 0
					if query_location == target_location:
						distance = 0
						global_location_distance_data_dict[query_location_id][query_model].append({ 'location_id': target_location_id, 'distance': 0 })
					else:
						# Find the corresponding model file of the query location in the target location
						target_model_file_path = FILE_PATH_PREFIX + target_location + " " + query_model + ".csv"
						target_model_df = pd.read_csv(target_model_file_path, header=None)
						target_model_df_copy = target_model_df.copy()
						del target_model_df[0]
						target_model_df = target_model_df.reset_index(drop=True)
						target_model_df_row_count = target_model_df.shape[0]
						target_model_df_column_count = target_model_df.shape[1]

						# Calculate the distance between the query location's model file and the target location's corresponding model file
						distance = self.get_the_distance_value(query_model_df, target_model_df, query_model_df_row_count, query_model, util)

						global_location_distance_data_dict[query_location_id][query_model].append({'location_id': target_location_id, 'distance': distance })

					# Set distance temporarily as 0 in the location_wise_distance_data_dict for this location
					if not location_wise_distance_data_dict.get(query_location_id).get(target_location_id):
						location_wise_distance_data_dict[query_location_id][target_location_id] = 0
				
				# At this state, we have gone over every target location with the corresponding model file from the query location.
				# Sort the model based location list of distances based on distance from the location
				sorted_list = sorted(global_location_distance_data_dict[query_location_id][query_model], key=lambda k: k['distance'])
				global_location_distance_data_dict[query_location_id][query_model].clear()
				global_location_distance_data_dict[query_location_id][query_model] = sorted_list
				# Repeat the loop, do it for every model file of the query location

			location_data_dict = global_location_distance_data_dict[query_location_id]

			# Compute the ranking of similar locations for the query location
			for curr_model, distance_list in location_data_dict.items():
				for index, curr_location_distance_data in enumerate(distance_list):
					curr_location_id = curr_location_distance_data['location_id']
					curr_val = location_wise_distance_data_dict[query_location_id][curr_location_id]
					location_wise_distance_data_dict[query_location_id][curr_location_id] = curr_val + index
			for l_id, dist in location_wise_distance_data_dict[query_location_id].items():
				similarity_matrix[query_location_id - 1][l_id - 1] = dist
			# Add this to similarity matrix

		print(similarity_matrix)

		# Generate CSVs of the current similarity matrix (given by distances derived from the ranks of individual models)

		# df = pd.DataFrame(similarity_matrix)
		# loc_list = []
		# for i in range(1,31):
		# 	loc_list.append(location_id_to_title_map[str(i)])

		# # Generate the distance datrix as CSV
		# df.to_csv('./generated_data/distance_matrix_vd_minmax.csv', encoding='utf-8', header=None, index=False)
		# df.to_csv('./generated_data/distance_matrix_vd_minmax_descriptive.csv', encoding='utf-8', header=loc_list, index=loc_list)

		# Convert distance score to similarity score
		converted_similarity_matrix = similarity_matrix
		for row in range(len(converted_similarity_matrix)):
			for col in range(len(converted_similarity_matrix[0])):
				# In the dev set case, it scales distance score that ranges from 0-290 in the computation to a similarity score ranging from 0-1
				converted_similarity_matrix[row][col] = ((float)(MAX_SCORE - converted_similarity_matrix[row][col])/MAX_SCORE)

		# Generate the similarity matrix as CSV if needed
		# df = pd.DataFrame(converted_similarity_matrix)
		# df.to_csv('./generated_data/similarity_matrix_vd_minmax.csv', encoding='utf-8', header=None, index=False)
		# df.to_csv('./generated_data/similarity_matrix_vd_descriptive.csv', encoding='utf-8')

		# Apply SVD on the data
		U, S, Vt = numpy.linalg.svd(converted_similarity_matrix)

		# {
		#  <location_id>: [{'Location Name': <>, 'Weight': <>}, {'Location Name': <>, 'Weight': <>}, ...],
		#  <location_id>: [{'Location Name': <>, 'Weight': <>}, {'Location Name': <>, 'Weight': <>}, ...],
		#  ...
		# }
		semantic_data_dict = {}
		for arr_index, arr in enumerate(Vt[:k, :]):
			if not semantic_data_dict.get(arr_index+1):
				semantic_data_dict[arr_index+1] = []

			for index, element in enumerate(arr):
				semantic_data_dict[arr_index+1].append({ 'Location Name': location_id_to_title_map[str(index+1)], 'Weight': element })

			# Sort the list based on the weight attribute
			sorted_list = sorted(semantic_data_dict[arr_index+1], key=itemgetter('Weight'), reverse=True)
			semantic_data_dict[arr_index+1].clear()
			semantic_data_dict[arr_index+1] = sorted_list

			# Print the latent semantic as location name-weight pairs sorted in decreasing order of weights
			print('Latent Semantic: ', arr_index+1)
			for idx, data in enumerate(sorted_list):
				print('\tLocation Name: ', semantic_data_dict[arr_index+1][idx]['Location Name'], '| Weight: ', semantic_data_dict[arr_index+1][idx]['Weight'])

if __name__ == '__main__':
	task6 = Task6()
	task6.runner()