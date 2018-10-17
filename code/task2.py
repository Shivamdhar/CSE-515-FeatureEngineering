'''
This module is the program for task 3. 
'''
from data_extractor import DataExtractor
import numpy as np
from scipy import spatial
from . import TxtTermStructure
from util import Util

class Task2(object):
    def __init__(self):
        self.ut = Util()
        self.data_extractor = DataExtractor()


    def get_terms_and_weight_vec(self):
        all_terms = []
        #Get the vector of all terms for a given user
        for k,v in self.user_dict.items():
            ind_term_vec = []
            for x in f.get_terms(k):
                y = x.replace('"','')
                ind_term_vec.append(y)
            all_terms.append(ind_term_vec)


        all_terms = []
        all_weights = []
        #Get the vector of all terms for a given user
        for k,v in self.master_dict.items():
            ind_term_vec = []
            ind_weight_vec = [texDescriptor.tfidf for texDescriptor in self.master_dict[k]]
            all_weights.append(ind_weight_vec)











    ''' Method: image-image and image-location similarity'''
    def calculate_similarity(self, k_semantics, image_position, array_of_all_images, array_location_vector):
        vector_of_input_image = k_semantics[image_position]
        similarity_score_images = []

        #Computing similarity between vector of input image and all the other vectors in k_semantics matrix
        for vector in k_semantics:
            result = spatial.distance.euclidean(vector_of_input_image,vector)
            result = 1 / (1 + result)
            similarity_score_images.append(result)

        #Storing all the image IDs and its score with given input image ID
        image_and_score = []
        for i in range(len(array_of_all_images)):
            image_and_score.append([array_of_all_images[i],similarity_score_images[i]])

        #Sorting on the basis of score and printing top 5 images across all locations
        sorted_sim_vector = sorted(image_and_score,key = lambda x:x[1],reverse = True) #sorting the similarity vector
        print("5 most similar images with matching score is :")
        print(sorted_sim_vector[:5])

        ''' The start index and end index for a location is used, the image to image scores
            for that location is sorted and the top value is stored for representing that location.
            The top values of all locations are sorted and the top 5 locations are printed. '''
        loc_img_score = []
        top_value = []
        for key in array_location_vector:
            start_index = array_location_vector[key][0]
            end_index = array_location_vector[key][1]
            top_value = sorted(similarity_score_images[start_index:end_index + 1],key = lambda x:x,reverse = True)[0]
            mapping = self.data_extractor.location_mapping()
            for loc_id,location_name in mapping.items():
                if(key == location_name):
                    location_id = loc_id
            loc_img_score.append([location_id,key,top_value])

        #Sorting on basis of score and printing top 5 locations
        top_locations = sorted(loc_img_score,key = lambda x:x[2],reverse = True)[:5]
        print("5 most similar locations with matching score is :")
        print(top_locations)

    '''
    Method: runner implemented for all the tasks, takes user input, runs dimensionality reduction algorithm, prints
    latent semantics and computes image-image and image-location similarity using the latent semantics.
    '''
    def runner(self):
        k = input("Enter the value of k :")
        #input for user/location/image id
        entity_id = input("Enter image ID : ")

        # array_of_all_images, image_input_array, image_position, \
        # array_location_vector = self.data_extractor.prepare_dataset_for_task3(model, image_id)

        # algo_choice = input("Enter the Algorithm: ")

        # algorithms = { "SVD": self.ut.dim_reduce_SVD, "PCA": self.ut.dim_reduce_PCA , "LDA": self.ut.dim_reduce_LDA}

        self.txt_term_structure = TxtTermStructure()

        txt_term_structure.get_desc_txt_data(constants.TEXT_DESCRIPTORS_PATH+'devset_textTermsPerUser.txt', 'users')

        self.master_dict = txt_term_structure.master_dict

        # get k semantics from the task1.
        k_semantics = []

        print(k_semantics)

        self.calculate_similarity(k_semantics)