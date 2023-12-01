#helper for deep learning training loop
import torch
import random
import numpy as np

def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True


##window must be less than num_train_batch #<todo> bug
class MyMeter():
	def __init__(self, value, window=100):
		length = len(value)
		self.length = length
		self.window = window
		self.value  = value
		self.accumate = [0]*length
		self.count  = 0
		self.history=[]

	def __getitem__(self, index):
		return self.value[index]

	def step(self, value):
		#print(self.count)
		self.history.append(value)
		for i in range(len(value)):
			self.accumate[i] += value[i]
		self.count +=1

		if self.count%self.window==0:
			for i in range(self.length):
				self.value[i] = self.accumate[i]/self.count
			self.accumate = [0] * self.length
			self.count = 0