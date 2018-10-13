'''
This module is a driver program which selects a particular task. 
'''
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from task4 import Task_num_4
from task5 import Task5
from util import Util

class Driver(object):

	def input_task_num(self):
		task_num = int(input("Enter the Task no.: "))
		self.select_task(task_num)

	def select_task(self, task_num):
		# Plugin class names for each task here
		tasks = { 1: "", 2: "", 3: "", 4: Task_num_4(), 5: Task5(), 6: "", 7: "" }

		# Have a runner method in all the task classes
		tasks.get(task_num).runner()

flag = True
while(flag):
	choice = int(input("Enter your choice:\t1) Compute Similarity\t2) Exit\n"))
	try:
		if choice == 2:
			flag = False
		else:
			t = Driver()
			t.input_task_num()
	except Exception as e:
		print("Exception encountered: ", str(type(e)) + "::" + str(e.args))

