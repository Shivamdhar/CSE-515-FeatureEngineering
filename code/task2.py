"""
This module is the program for task 2.
"""
import constants
from data_extractor import DataExtractor
import numpy as np
import scipy
from scipy import spatial
from sklearn.decomposition import LatentDirichletAllocation
from sparsesvd import sparsesvd
from task1 import Task1
from textual_descriptor_processor import TxtTermStructure
from util import Util


class Task2(object):
	"""
	This module is responsible for finding similarity between entities based on the latent semantics
	computed by PCA/LDA/SVD on original vector space (user/location/image)
	"""
	def __init__(self):
		self.ut = Util()
		self.data_extractor = DataExtractor()
		self.mapping = self.data_extractor.location_mapping()

	def calculate_similarity(self, input_vector, k_semantics_map, entity_type):
		'''
		Method : calculate_similarity computes compute the similarity 
		matrix for the given entity id and given the latent semantics vector
		in form of weights for given entity type
		'''
		#k_semantics = [v for k,v in k_semantics_map]

		#input_vector = k_semantics_map.get(entity_id,[])

		similarity_data = []

		# if entity_type == constants.LOCATION_TEXT:
		# 	mapping = self.data_extractor.location_mapping()

		for key,value in k_semantics_map.items():
			result = self.ut.compute_euclidean_distance(input_vector,value)
			result = 1 / (1 + result)
			# if entity_type == constants.LOCATION_TEXT:
			# 	key = mapping[key]
			if entity_type == constants.IMAGE_TEXT:
				key = int(key)
			similarity_data.append((key,result))

		return similarity_data

	def get_k_semantics_map(self,entity_data,k_semantics):
		"""
		Method : Returns the k semantics map while linking the entity id from original data 
		with new latent semantics
		entity_data : user_term matrix or image_term matrix or location_term matrix
		k_semantics : new reduced feature space after projecting the original data points with
		dimension o x k
		"""
		entity_ids = list(entity_data.data.master_dict.keys())
		k_semantics_map = {}
		for entity_id,value in zip(entity_ids,k_semantics):
			k_semantics_map[entity_id] = value

		return k_semantics_map

	def top_5(self, similarity_data):
		"""
		Method: Prints the top5 similar entities (users/images/locations) with respect to input
		entity
		similarity_data: List of objects containing entity_id and the similarity score between
		respective entity
		"""
		print(sorted(similarity_data, key=lambda x: x[1], reverse=True)[:5])
		pass

	def dim_reduce_SVD(self,document_term_matrix,k,pca=False):
		"""
		Method: Returns the left factor, Sigma and right factor matrix using SVD and PCA if pca flag
		is True
		document_term_matrix : Input data matrix
		k  : number of hidden concepts
		"""
		document_term_matrix = self.ut.convert_list_to_numpyarray(document_term_matrix)
		if pca:
			document_term_matrix = np.cov(document_term_matrix.T)
		document_term_sparse_matrix = scipy.sparse.csc_matrix(document_term_matrix)
		#print(document_term_sparse_matrix)
		U,S,Vt = sparsesvd(document_term_sparse_matrix,k)

		#Projection of objects along hidden concepts
		U = document_term_sparse_matrix @ Vt.T

		# original sigma is linear array of k components, so we need to construct a diagonal matrix
		S = np.diag(S)

		#since U is actually kxo, so we take transpose
		return U,S,Vt
		pass

	def dim_reduce_LDA(self,document_term_matrix,k):
		"""
		Method: Returns the left factor, Sigma and right factor matrix using LDA
		document_term_matrix : Input data matrix
		k  : number of hidden concepts
		"""
		lda = LatentDirichletAllocation(n_components=int(k),max_iter=5, 
				learning_method='online', random_state=0)
		document_topic_matrix = lda.fit_transform(document_term_matrix)

		term_topic_matrix = lda.components_

		#Getting the feature vs feature matrix
		topic_topic_matrix = term_topic_matrix @ term_topic_matrix.T

		#Projection of original objects along hidden topics
		transformed_document_topic_matrix = document_topic_matrix @ topic_topic_matrix

		return transformed_document_topic_matrix,topic_topic_matrix,term_topic_matrix

	def get_projected_query_vector(self,input_vector,v_matrix,sigma_matrix):
		"""
		Method:  Returns the projected query vector onto given latent semantic space
		input_vector : input_query from original data matrix form
		v_matrix : feature vs k concepts matrix
		sigma_matrix : core_matrix
		"""
		projected_query_vector = []

		#print("pre diagonal S shape",sigma_matrix.shape)

		#diagonal_sigma_matrix = np.diag(sigma_matrix)

		# print("IP shape",input_vector.shape)
		# print("V shape",v_matrix.shape)
		# print("S shape",sigma_matrix.shape)

		projected_query_vector = input_vector.T @ v_matrix.T @ np.linalg.inv(sigma_matrix)

		return projected_query_vector

	def get_document_term_matrix(self,entity_data):
		"""
		Method : Get the document term matrix for the given entity using the global dictionary of 
		terms.
		entity_data : entity_data matrix
		"""
		global_tag_set = entity_data.get_global_tag_set()
		global_tag_dict = entity_data.convert_dict_from_set(global_tag_set)
		master_matrix = entity_data.create_master_matrix(global_tag_dict)

		return master_matrix

	def get_similar_entities(self,user_term_matrix,image_term_matrix,
				location_term_matrix,user_S_matrix,user_vt_matrix,image_S_matrix,image_vt_matrix,
					location_S_matrix, location_vt_matrix,user_id=None,image_id=None,location_id=None):
		"""
		Method: Get the similar users, images and locations by projecting the given query vector onto 
		other latent semantic space for all user, image and location entity
		
		user_term_matrix: user document_term matrix
		image_term_matrix: image document_term matrix
		location_term_matrix: location document_term matrix
		user_S_matrix :  sigma matrix after decomposing user data matrix
		image_S_matrix :  sigma matrix after decomposing image data matrix
		location_S_matrix :  sigma matrix after decomposing location data matrix

		user_vt_matrix :  feature vs concepts matrix after decomposing user data matrix
		image_vt_matrix :  feature vs concepts matrix after decomposing image data matrix
		location_vt_matrix :  feature vs concepts matrix after decomposing location data matrix
		"""

		if user_id:
			#If the input is user_id
			user_input_vector = self.user_semantics_map[user_id]

			#For similar user id we can directly use the user U matrix without projecting
			similar_users = self.calculate_similarity(user_input_vector,self.user_semantics_map,constants.USER_TEXT)

			original_user_input_vector = self.ut.convert_list_to_numpyarray(user_term_matrix[self.user_index])

			user_projected_query_vector_image = self.get_projected_query_vector(original_user_input_vector,image_vt_matrix,image_S_matrix)
			user_projected_query_vector_location = self.get_projected_query_vector(original_user_input_vector,location_vt_matrix,location_S_matrix)

			similar_images = self.calculate_similarity(user_projected_query_vector_image,self.image_semantics_map,constants.IMAGE_TEXT)
			similar_locations = self.calculate_similarity(user_projected_query_vector_location,self.location_semantics_map,constants.LOCATION_TEXT)

		elif image_id:
			#Given image id, computing the top 5 related images ,users and locations
			image_input_vector = self.image_semantics_map[image_id]

			#For similar user id we can directly use the user U matrix without projecting
			similar_images = self.calculate_similarity(image_input_vector,self.image_semantics_map,constants.IMAGE_TEXT)

			original_image_input_vector = self.ut.convert_list_to_numpyarray(image_term_matrix[self.image_index])

			image_projected_query_vector_user = self.get_projected_query_vector(original_image_input_vector,user_vt_matrix,user_S_matrix)
			image_projected_query_vector_location = self.get_projected_query_vector(original_image_input_vector,location_vt_matrix,location_S_matrix)

			similar_users = self.calculate_similarity(image_projected_query_vector_user,self.user_semantics_map,constants.USER_TEXT)
			similar_locations = self.calculate_similarity(image_projected_query_vector_location,self.location_semantics_map,constants.LOCATION_TEXT)

		elif location_id:
			#Given location id, computing the top 5 related locations,users and images
			location_input_vector = self.location_semantics_map[self.mapping[location_id]]

			#For similar user id we can directly use the user U matrix without projecting
			similar_locations = self.calculate_similarity(location_input_vector,self.location_semantics_map,constants.LOCATION_TEXT)
			#self.top_5(similar_locations)

			original_location_input_vector = self.ut.convert_list_to_numpyarray(location_term_matrix[self.location_index])

			location_projected_query_vector_image = self.get_projected_query_vector(original_location_input_vector,image_vt_matrix,image_S_matrix)
			location_projected_query_vector_user = self.get_projected_query_vector(original_location_input_vector,user_vt_matrix,user_S_matrix)

			# print("Shape1",location_projected_query_vector_image.shape)
			# print("Shape2",location_projected_query_vector_user.shape)

			similar_images = self.calculate_similarity(location_projected_query_vector_image,self.image_semantics_map,constants.IMAGE_TEXT)
			similar_users = self.calculate_similarity(location_projected_query_vector_user,self.user_semantics_map,constants.USER_TEXT)

		print("Top 5 related users are")
		self.top_5(similar_users)
		print("Top 5 related images are")
		self.top_5(similar_images)
		print("Top 5 related locations are")
		self.top_5(similar_locations)
		pass

	def get_all_latent_semantics_map(self,user_data,image_data,location_data,user_u_matrix,
				image_u_matrix,location_u_matrix):

		user_semantics_map = self.get_k_semantics_map(user_data,user_u_matrix)
		image_semantics_map = self.get_k_semantics_map(image_data,image_u_matrix)
		location_semantics_map = self.get_k_semantics_map(location_data,location_u_matrix)

		return user_semantics_map,image_semantics_map,location_semantics_map

	def runner(self):
		"""
		Method: runner implemented for all the tasks, 
		takes user input for type of entity from list of User,Image and Location
		and respective entity_id
		Displays the top 5 entities with respect to input entity 
		using the latent semantics obtained from task1 for respective entity vector space
		"""
		#k = input("Enter the value of k :")
		k = input("Enter the value of k :")

		# user_id = input("Enter the user id: ")
		# image_id = input("Enter the image id: ")
		# location_id = input("Enter the location id: ")
		
		algo_choice = input("Enter the Algorithm: ")

		entity_index = int(input("Choose the entity id \t1) User \t2)Image \t3)Location.: "))

		user_id,image_id,location_id = None,None,None

		if entity_index == 1:
			self.entity_type = constants.USER_TEXT
			user_id = input("Enter the user id: ")

		elif entity_index == 2:
			self.entity_type = constants.IMAGE_TEXT
			image_id = input("Enter the image id: ")

		elif entity_index == 3:
			self.entity_type = constants.LOCATION_TEXT
			location_id = input("Enter the location id: ")

		"""
		Get the document term matrix for users,images and locations from task1
		"""
		user_data = Task1()
		user_data.load_data_per_entity(constants.USER_TEXT)
		user_term_matrix = self.get_document_term_matrix(user_data)

		image_data = Task1()
		image_data.load_data_per_entity(constants.IMAGE_TEXT)
		image_term_matrix = self.get_document_term_matrix(image_data)

		location_data = Task1()
		location_data.load_data_per_entity(constants.LOCATION_TEXT)
		location_term_matrix = self.get_document_term_matrix(location_data)

		if self.entity_type == constants.USER_TEXT:
			try:
				self.user_index = list(user_data.data.master_dict.keys()).index(user_id)
			except ValueError:
				raise ValueError(constants.USER_ID_KEY_ERROR)
			pass
		elif self.entity_type == constants.IMAGE_TEXT:
			try:
				self.image_index = list(image_data.data.master_dict.keys()).index(image_id)
			except ValueError:
				raise ValueError(constants.IMAGE_ID_KEY_ERROR)
			pass
		elif self.entity_type == constants.LOCATION_TEXT:
			try:
				input_location = self.mapping[location_id]
				self.location_index = list(location_data.data.master_dict.keys()).index(input_location)
			except ValueError:
				raise ValueError(constants.LOCATION_ID_KEY_ERROR)
			pass
		
		if algo_choice == 'SVD' or algo_choice == 'PCA':
			# user_term_sparse_matrix = scipy.sparse.csc_matrix(user_term_matrix)
			# print(user_term_sparse_matrix)
			pca = False
			if algo_choice == 'PCA':
				pca = True
			"""
			Decompose the original document term matrix into U,S and Vt using SVD
			For PCA we pass pca flag to indicate the passing of covariance matrix in the SVD method.
			"""
			user_u_matrix, user_S_matrix, user_vt_matrix = self.dim_reduce_SVD(
				user_term_matrix,k,pca)
			image_u_matrix, image_S_matrix, image_vt_matrix = self.dim_reduce_SVD(
				image_term_matrix,k,pca)
			location_u_matrix, location_S_matrix, location_vt_matrix = self.dim_reduce_SVD(
				location_term_matrix,k,pca)

			"""
			Get the latent semantics for users, images and locations
			"""
			user_semantics_map, image_semantics_map,location_semantics_map = \
					self.get_all_latent_semantics_map(user_data,image_data,location_data,
						user_u_matrix,image_u_matrix,location_u_matrix)

			self.user_semantics_map = user_semantics_map
			self.image_semantics_map = image_semantics_map
			self.location_semantics_map = location_semantics_map

			"""
			Get the similar cross entities given a entity id. eg userid -> similar users, images,
			and locations,  imageid -> similar images, locations and users.
			"""
			self.get_similar_entities(user_term_matrix,image_term_matrix,
				location_term_matrix,user_S_matrix,user_vt_matrix,image_S_matrix,image_vt_matrix,
					location_S_matrix, location_vt_matrix,user_id,image_id,location_id)

		elif algo_choice == 'LDA':
			"""
			Decompose the original document term matrix into U,S and Vt using LDA
			"""
			user_u_matrix, user_S_matrix, user_vt_matrix = self.dim_reduce_LDA(
				user_term_matrix,k)
			image_u_matrix, image_S_matrix, image_vt_matrix = self.dim_reduce_LDA(
				image_term_matrix,k)
			location_u_matrix, location_S_matrix, location_vt_matrix = self.dim_reduce_LDA(
				location_term_matrix,k)

			# print("dfdF",location_u_matrix.shape)
			# print("sdsdsd",user_u_matrix.shape)
			# print("sdsfdf",image_u_matrix.shape)

			user_semantics_map, image_semantics_map,location_semantics_map = \
					self.get_all_latent_semantics_map(user_data,image_data,location_data,
						user_u_matrix,image_u_matrix,location_u_matrix)

			self.user_semantics_map = user_semantics_map
			self.image_semantics_map = image_semantics_map
			self.location_semantics_map = location_semantics_map

			self.get_similar_entities(user_term_matrix,image_term_matrix,
				location_term_matrix,user_S_matrix,user_vt_matrix,image_S_matrix,image_vt_matrix,
					location_S_matrix, location_vt_matrix,user_id,image_id,location_id)


