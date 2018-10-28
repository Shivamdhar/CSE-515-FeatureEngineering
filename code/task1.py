"""
This module is the program for task 1 phase 2. 
"""
import constants
from util import Util
from textual_descriptor_processor import TxtTermStructure

class Task1(object):
	def __init__(self):
		self.data = TxtTermStructure()
		self.ut = Util()
			
	def runner(self):
		"""
		Method: runner implemented for all the tasks, takes user input, runs dimensionality reduction algorithm, prints
		latent semantics
		"""

		self.select_term_vector_choice()
		global_tag_set = self.get_global_tag_set()
		global_tag_dict = self.convert_dict_from_set(global_tag_set)
		master_matrix = self.create_master_matrix(global_tag_dict)
		algo_choice = input("Enter the Dim. reduction Algorithm: ")
		k = input("Enter the value of k :")
		algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA_nonscaler, \
						"LDA": self.ut.dim_reduce_LDA }            
		k_semantics = algorithms.get(algo_choice)(master_matrix, k)
		print(k_semantics)
		
	def select_term_vector_choice(self):
		"""
		lets user choose among :user-term vector space, image-term vector space and location-term vector space.
		"""
		term_vector_space_choice = int(input("Enter the term-vector space: \t1)user \t2)image \t3)location.: "))
		if term_vector_space_choice == 1: 
			self.data.load_users_data() 
		if term_vector_space_choice == 2: 
			self.data.load_image_data()
		if term_vector_space_choice == 3:
			self.data.load_location_data()

	def load_data_per_entity(self,entity_type):
		if entity_type == constants.USER_TEXT:
			self.data.load_users_data()
		if entity_type == constants.IMAGE_TEXT:
			self.data.load_image_data()
		if entity_type == constants.LOCATION_TEXT:
			self.data.load_location_data()

	def get_global_tag_set(self):
		"""
		creates set of global tags 
		"""
		union_tag_set = set()
		for key in self.data.master_dict:
			union_tag_set = union_tag_set.union(self.data.get_terms(key))

		return union_tag_set

	def convert_dict_from_set(self, data_set):
		"""
		this method initializes set of all the tags equal to zero. Later it can be assigned with the 
		tag's corresponding TF-IDF score as per each user's usage of that tag against the global tag set.
		"""
		return {x:0 for x in data_set}

	def merge_two_dicts(self, x, y):
		"""
		assigns tags' weight for each user based local dictionary against global dictionary of tags
		"""
		z = x.copy()   # start with x's keys and values
		z.update(y)    # modifies z with y's keys and values & returns None

		return z

	def create_master_matrix(self, master_term_dict):
		"""
		creates high dimensional matrix with each row containing TF-IDF score for each user as per global dictionary
		"""
		master_list = []
		list_master_matrix = []

		for data_row in self.data.master_dict:
			row_term_dict = self.data.get_term_tf_idf(data_row)
			global_row_dict = self.merge_two_dicts(master_term_dict, row_term_dict)
			list_master_matrix.append(list(global_row_dict.values()))

		return list_master_matrix
		
	   