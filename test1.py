import numpy as np 



class my_generator(object):
	"""docstring for my_generator"""
	def __init__(self, X):
		super(my_generator, self).__init__()
		self.X = X
		self.N = len(self.X)
		self.current = 0
	def next(self, batch_size=3):
		## assert self.N > batch size !!
		if self.current + batch_size < self.N:
			A = self.X[self.current: self.current + batch_size]
			self.current = self.current + batch_size
			return A
		else:
			res = self.N - self.current 
			A = self.X[self.current:]
			B = self.X[0:(batch_size - res)]
			
			self.current = batch_size - res
			return np.vstack([A, B])



if __name__=="__main__":
	print("Start")
	a = np.arange(10).reshape(-1,1)
	A = my_generator(a)
	import ipdb;ipdb.set_trace()