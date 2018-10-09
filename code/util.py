'''
This module contains all functions used throughout the codebase. 
'''
import os 
import numpy as np
from scipy.spatial.distance import cosine as cs
import xml.etree.ElementTree as et
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy import spatial

class Util(object):

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

	def create_dataset(self, mapping, model, LocationID):
		folder = "../dataset/visual_descriptors/"
		location_names = list(mapping.values())
		fileList = []
		"""fileList contains list of tuples which are of the form [('location file path', 'location') for all other 
		locations other than the input location"""
		x = len(mapping)
		for i in range(0,x):
			if i != (int(LocationID)-1):
				fileList.append((folder + location_names[i] + " " + model + ".csv", location_names[i]))
		return fileList

	def append_givenloc_to_list(self, mapping, model, LocationID, fileList):
		folder = "../dataset/visual_descriptors/"	
		location_list_indices = {}  
		inputImageList = []     
		givenFile = folder + mapping[LocationID] + " " + model + ".csv"
		with open (givenFile) as f:
			givenfileData = (f.read()).split("\n")[:-1]
		for each_given_row in givenfileData:
			inputImageList.append(each_given_row.split(",")[1:])
		location_list_indices.update({mapping[LocationID]:[0,len(inputImageList)]})

		start = len(inputImageList)
		for each in fileList:
			eachFile = each[0]
			title = each[1]
			with open (eachFile) as eFile:
				fileData = (eFile.read()).split("\n")[:-1]#read data from each of the file in fileList
			for eachRow in fileData:
				inputImageList.append(eachRow.split(",")[1:])
			location_list_indices.update({title:[start,len(inputImageList)]})
			start = len(inputImageList)

		return inputImageList, location_list_indices

	def convert_list_to_numpyarray(self, inputImageList):	    
		inputImageArr = np.array(inputImageList, dtype=np.float64)
		return inputImageArr
