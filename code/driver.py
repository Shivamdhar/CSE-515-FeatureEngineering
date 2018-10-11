'''
This module is a driver program which selects a particular task. 
'''
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from task4 import Task_num_4
from util import Util

class Driver(object):

	def input_task_num(self):
		task_num = input("Enter the Task no.: ")
		self.select_task(task_num)

	def select_task(self, task_num):
		#create the objects of each task here
		# ToDo: change this to generic object creation
		t4 = Task_num_4()

		if(task_num == '1'):
			pass
		elif(task_num == '2'):
			pass
		elif(task_num == '3'):
			pass
		elif(task_num == '4'):
			t4.task4()
		elif(task_num == '5'):
			pass
		elif(task_num == '6'):
			pass
		elif(task_num == '7'):
			pass
		else:
			print("Incorrect Task No. Please Enter correct No.")
	
t = Driver()
t.input_task_num()
		
