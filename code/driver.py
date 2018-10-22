"""
This module is a driver program which selects a particular task. 
"""
import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from task1 import Task1
from task2 import Task2
from task3 import Task3
from task4 import Task4
from task5 import Task5
from task7 import Task7
from task_6_textual_descriptors import Task6
from util import Util

class Driver(object):

	def input_task_num(self):
		task_num = int(input("Enter the Task no.: "))
		self.select_task(task_num)

	def select_task(self, task_num):
		# Plugin class names for each task here
		tasks = { 1: Task1(), 2: Task2(), 3: Task3(), 4: Task4(), 5: Task5(), 6: Task6(), 7: Task7() }
		# Have a runner method in all the task classes
		tasks.get(task_num).runner()

flag = True
while(flag):
	choice = int(input("Enter your choice:\t1) Compute Similarity\t2) Exit\n"))
	
	if choice == 2:
		flag = False
	else:
		t = Driver()
		t.input_task_num()